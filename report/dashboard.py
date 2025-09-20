from fasthtml.common import *
import matplotlib.pyplot as plt
from employee_events import QueryBase, Employee, Team

# --- Robust imports: support both "module" and "script" execution ---
try:
    # When running as a module (uvicorn report.dashboard:app)
    from .utils import load_model
    from .base_components import (
        Dropdown,
        BaseComponent,
        Radio,
        MatplotlibViz,
        DataTable
    )
    from .combined_components import FormGroup, CombinedComponent
except ImportError:
    # When running as a script (python report/dashboard.py)
    from utils import load_model
    from base_components import (
        Dropdown,
        BaseComponent,
        Radio,
        MatplotlibViz,
        DataTable
    )
    from combined_components import FormGroup, CombinedComponent


# ---------------------------
# Dropdown for Employee/Team
# ---------------------------
class ReportDropdown(Dropdown):
    def build_component(self, entity_id, model):
        # Label shows the current profile type
        self.label = getattr(model, "name", "selection")
        return super().build_component(entity_id, model)

    def component_data(self, entity_id, model):
        # Expect list of (label, value) tuples
        try:
            rows = model.names() or []
        except Exception:
            rows = []
        return rows


# ---------------------------
# Header (dynamic title)
# ---------------------------
class Header(BaseComponent):
    def build_component(self, entity_id, model):
        title = f"{getattr(model, 'name', 'Report').capitalize()} Performance"
        return H1(title)


# ---------------------------
# Line Chart (cumulative events)
# ---------------------------
class LineChart(MatplotlibViz):
    def visualization(self, asset_id, model):
        import pandas as pd
        try:
            df = model.event_counts(asset_id)
        except Exception:
            df = pd.DataFrame(columns=["event_date", "positive_events", "negative_events"])

        if df is None or df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=16)
            ax.axis("off")
            return fig

        df = df.fillna(0)
        df = df.set_index("event_date").sort_index()

        cum = df[["positive_events", "negative_events"]].cumsum()
        cum.columns = ["Positive", "Negative"]

        fig, ax = plt.subplots()
        cum.plot(ax=ax)

        # Use default styling (no kwargs)
        self.set_axis_styling(ax)
        ax.set_title("Cumulative Productivity")
        ax.set_xlabel("Date")
        ax.set_ylabel("Events (cumulative)")
        return fig


# ---------------------------
# Bar Chart (ML risk with color scale)
# ---------------------------
class BarChart(MatplotlibViz):
    predictor = load_model()

    def visualization(self, asset_id, model):
        import numpy as np

        # Get features for prediction
        try:
            X = model.model_data(asset_id)
        except Exception:
            X = None

        # Predict probability (robust to regressors)
        p = np.array([0.0])
        if X is not None:
            try:
                proba = self.predictor.predict_proba(X)
                p = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.reshape(-1)
            except Exception:
                try:
                    p = np.array(self.predictor.predict(X)).reshape(-1)
                except Exception:
                    p = np.array([0.0])

        if getattr(model, "name", "") == "team":
            pred = float(np.mean(p)) if p.size else 0.0
        else:
            pred = float(p[0]) if p.size else 0.0

        fig, ax = plt.subplots()
        cmap = plt.cm.get_cmap("RdYlGn_r")  # green -> red
        color = cmap(max(0.0, min(1.0, pred)))

        ax.barh([''], [pred], color=[color])
        ax.set_xlim(0, 1)
        ax.set_title('Predicted Recruitment Risk', fontsize=20)

        ax.text(
            pred,
            0,
            f"{pred*100:.1f}%",
            va="center",
            ha="left" if pred < 0.9 else "right",
            fontsize=12
        )

        # Use default styling (no kwargs)
        self.set_axis_styling(ax)
        return fig


# ---------------------------
# Visualizations container
# ---------------------------
class Visualizations(CombinedComponent):
    children = [LineChart(), BarChart()]
    outer_div_type = Div(cls='grid')


# ---------------------------
# Notes Table (DB data)
# ---------------------------
class NotesTable(DataTable):
    def component_data(self, entity_id, model):
        return model.notes(entity_id)


# ---------------------------
# Top filters (radio + dropdown)
# ---------------------------
class DashboardFilters(FormGroup):
    id = "top-filters"
    action = "/update_data"
    method = "POST"

    children = [
        Radio(
            values=["Employee", "Team"],
            name='profile_type',
            hx_get='/update_dropdown',
            hx_target='#selector'
        ),
        ReportDropdown(
            id="selector",
            name="user-selection"
        )
    ]


# ---------------------------
# Page composition
# ---------------------------
class Report(CombinedComponent):
    children = [Header(), DashboardFilters(), Visualizations(), NotesTable()]


# ---------------------------
# App + routes
# ---------------------------
app = FastHTML()
report = Report()


@app.get("/")
def index():
    # Default to Employee 1
    return report(1, Employee())


@app.get("/employee/{id:str}")
def employee_page(id: str):
    return report(int(id), Employee())


@app.get("/team/{id:str}")
def team_page(id: str):
    return report(int(id), Team())


# --- HTMX update endpoints (unchanged) ---
@app.get('/update_dropdown{r}')
def update_dropdown(r):
    dropdown = DashboardFilters.children[1]
    print('PARAM', r.query_params['profile_type'])
    if r.query_params['profile_type'] == 'Team':
        return dropdown(None, Team())
    elif r.query_params['profile_type'] == 'Employee':
        return dropdown(None, Employee())


@app.post('/update_data')
async def update_data(r):
    from fasthtml.common import RedirectResponse
    data = await r.form()
    profile_type = data._dict['profile_type']
    id = data._dict['user-selection']
    if profile_type == 'Employee':
        return RedirectResponse(f"/employee/{id}", status_code=303)
    elif profile_type == 'Team':
        return RedirectResponse(f"/team/{id}", status_code=303)


# Allow "python report/dashboard.py"
serve()
