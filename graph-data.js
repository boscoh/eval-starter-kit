// Performance data extracted from evals/results/consultant*.yaml files
// Each entry contains the basename, average elapsed_seconds, and standard deviation

const consultantPerformanceData = [
  {
    basename: "consultant-bedrock-nova-pro",
    service: "bedrock",
    model: "nova-pro",
    average: 2.3907617330551147,
    standardDeviation: 1.5022632179467799
  },
  {
    basename: "consultant-groq-llama-3.3-70b",
    service: "groq",
    model: "llama-3.3-70b",
    average: 0.5591460466384888,
    standardDeviation: 0.021206439709347397
  },
  {
    basename: "consultant-openai-gpt-4o",
    service: "openai",
    model: "gpt-4o",
    average: 3.2602646350860596,
    standardDeviation: 0.5573863395851761
  },
  {
    basename: "consultant-ollama-llama3.2",
    service: "ollama",
    model: "llama3.2",
    average: 26.01691746711731,
    standardDeviation: 2.47202460719358
  },
  {
    basename: "consultant",
    service: "unknown",
    model: "unknown",
    average: 11.625265836715698,
    standardDeviation: 0.06054276777414097
  }
];

// Plotly-formatted graphs for graph.html
window.graphs = [
  {
    id: "consultant-performance-graph",
    data: [
      {
        y: [
          "bedrock nova-pro",
          "groq llama-3.3-70b",
          "openai gpt-4o",
          // "ollama llama3.2"
        ],
        x: [
          2.3907617330551147,
          0.5591460466384888,
          3.2602646350860596,
          // 26.01691746711731
        ],
        error_x: {
          type: "data",
          array: [
            1.5022632179467799,
            0.021206439709347397,
            0.5573863395851761,
            // 2.47202460719358
          ],
          visible: true,
          color: "#444",
          thickness: 1,
          width: 6
        },
        type: "bar",
        orientation: "h",
        width: 0.7,
        marker: {
          color: ["#a8d5f7", "#ffd9a8", "#b3e6b3"]
        }
      }
    ],
    layout: {
      title: {
        text: "Consultant Query Performance by Service/Model",
        font: { size: 20 }
      },
      xaxis: {
        title: "Elapsed Time (seconds)"
      },
      yaxis: {
        title: "",
        ticklen: 10,
        tickcolor: "transparent"
      },
      bargap: 0.05,
      margin: { t: 80, b: 80, l: 250, r: 40 },
      height: 300
    }
  }
];

// Export for use in other modules (if using ES6 modules)
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    consultantPerformanceData,
    graphs: window.graphs
  };
}

