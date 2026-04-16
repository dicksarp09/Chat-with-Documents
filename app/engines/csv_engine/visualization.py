import logging
from typing import Dict, Any, List, Optional

from app.shared.llm_clients import get_llm_client
from app.shared.schemas import PlotSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisualizationGenerator:
    def __init__(self):
        self.llm = get_llm_client()
        self.max_plots = 3

    def generate_plots(
        self, query: str, results: List[Dict[str, Any]], schema: Dict[str, Any]
    ) -> List[PlotSpec]:
        logger.info("Generating visualization suggestions")

        columns = []
        if "columns" in schema:
            for col in schema["columns"]:
                if col.get("type") in ["numeric", "datetime"]:
                    columns.append(col["name"])

        if len(columns) < 1:
            return []

        prompt = f"""Based on the data analysis results, suggest appropriate visualizations.

QUERY: {query}

AVAILABLE COLUMNS: {", ".join(columns)}

PLOT TYPES AVAILABLE:
- bar: Compare categories
- line: Show trends over time
- scatter: Show relationships between variables
- histogram: Show distribution
- pie: Show proportions
- box: Show distribution and outliers

RULES:
1. Maximum 3 plots
2. Use only available columns
3. Choose appropriate plot types
4. Each plot must have x, y, and title

OUTPUT FORMAT (JSON only):
{{
    "plots": [
        {{
            "type": "bar|line|scatter|histogram|pie|box",
            "x": "column_name",
            "y": "column_name",
            "title": "descriptive title",
            "color_by": "optional_column"
        }}
    ]
}}

JSON OUTPUT:"""

        response = self.llm.generate_json(prompt)

        if not response or "plots" not in response:
            return self._fallback_plots(columns)

        plots = []
        for plot_data in response.get("plots", [])[: self.max_plots]:
            try:
                plot = PlotSpec(
                    type=plot_data.get("type", "bar"),
                    x=plot_data.get("x", columns[0]),
                    y=plot_data.get("y", columns[0]),
                    title=plot_data.get("title", "Chart"),
                    color_by=plot_data.get("color_by"),
                )

                if self._validate_plot(plot, columns):
                    plots.append(plot)
            except Exception as e:
                logger.warning(f"Invalid plot data: {e}")
                continue

        return plots[: self.max_plots]

    def _validate_plot(self, plot: PlotSpec, available_columns: List[str]) -> bool:
        if plot.x not in available_columns:
            return False
        if plot.y not in available_columns:
            return False
        if plot.color_by and plot.color_by not in available_columns:
            return False
        if plot.type not in ["bar", "line", "scatter", "histogram", "pie", "box"]:
            return False
        return True

    def _fallback_plots(self, columns: List[str]) -> List[PlotSpec]:
        if len(columns) >= 2:
            return [
                PlotSpec(type="bar", x=columns[0], y=columns[1], title="Distribution")
            ]
        return []

    def generate_plot_code(self, plot: PlotSpec, df_name: str = "df") -> str:
        if plot.type == "bar":
            return f'df.groupby("{plot.x}")["{plot.y}"].sum().plot(kind="bar", title="{plot.title}")'
        elif plot.type == "line":
            return f'df.plot(kind="line", x="{plot.x}", y="{plot.y}", title="{plot.title}")'
        elif plot.type == "scatter":
            return f'df.plot(kind="scatter", x="{plot.x}", y="{plot.y}", title="{plot.title}")'
        elif plot.type == "histogram":
            return f'df["{plot.x}"].plot(kind="hist", title="{plot.title}")'
        elif plot.type == "pie":
            return f'df.groupby("{plot.x}")["{plot.y}"].sum().plot(kind="pie", title="{plot.title}")'
        elif plot.type == "box":
            return f'df.boxplot(column="{plot.y}", by="{plot.x}", title="{plot.title}")'
        else:
            return (
                f'df.plot(kind="bar", x="{plot.x}", y="{plot.y}", title="{plot.title}")'
            )


def generate_visualizations(
    query: str, results: List[Dict[str, Any]], schema: Dict[str, Any]
) -> List[PlotSpec]:
    generator = VisualizationGenerator()
    return generator.generate_plots(query, results, schema)
