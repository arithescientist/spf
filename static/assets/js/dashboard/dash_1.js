try {

  Apex.tooltip = {
    theme: 'dark'
  }


fetch("static/assets/js/dashboard/pie.json").then(resp => resp.json())
    .then(_data => {

      var options = {
          chart: {
              type: 'donut',
              width: 500,
              height: 500
          },
          colors: ['#5c1ac3', '#e2a03f', '#e7515a', '#e2a03f'],
          dataLabels: {
            enabled: false
          },
          legend: {
              position: 'bottom',
              horizontalAlign: 'center',
                // width: 4,
              fontSize: '14px',
              // offsetY: 15,
              markers: {
                width: 10,
                height: 10,
              },
              itemMargin: {
                horizontal: 0,
                vertical: 25
              }
          },
       title: {
          text: 'Twitter Sentiment Analysis',
          align: 'center',
          floating: false,
          // offsetX: 100,
          offsetY: 15,
          style: {
            fontSize: '20px',
          color:  '#bfc9d4'
          },
        },
          plotOptions: {
            pie: {
            startAngle: 0,
            expandOnClick: true,
            offsetX: 0,
            offsetY: 0,
            customScale: 1,
            dataLabels: {
                offset: 0,
                minAngleToShowLabel: 10
            }, 
              donut: {
                size: '65%',
                background: 'transparent',
                labels: {
                  show: true,
                  name: {
                    show: true,
                    fontSize: '25px',
                    fontFamily: 'Nunito, sans-serif',
                    color: undefined,
                    offsetY: -10
                  },
                  value: {
                    show: true,
                    fontSize: '26px',
                    fontFamily: 'Nunito, sans-serif',
                    color: '#bfc9d4',
                    offsetY: 16,
                    formatter: function (val) {
                      return val
                    }
                  },
                  total: {
                    show: true,
                    showAlways: true,
                    fontSize: '10px',
                    fontWeight: 800,
                    label: 'Tweet Count',
                    color: '#888ea8',
                    formatter: function (w) {
                      return w.globals.seriesTotals.reduce( function(a, b) {
                        return a + b
                      }, 0)
                    }
                  }
                }
              }
            }
          },
          stroke: {
            show: true,
            width: 25,
            colors: '#0e1726'
          },
          series: _data.map(i => i.sizes),
          labels: _data.map(i => i.labels),
      };

     
      var chart = new ApexCharts(
          document.querySelector("#pie"),
          options
      );

      chart.render();
});

fetch("static/assets/js/dashboard/trends.json").then(resp => resp.json())
    .then(_data => {

      var options1 = {
        chart: {
          fontFamily: 'Nunito, sans-serif',
          height: 365,
          type: 'area',
          zoom: {
            type: 'x',
              enabled: true
          },
          dropShadow: {
            enabled: true,
            opacity: 0.3,
            blur: 5,
            left: -7,
            top: 22
          },
          toolbar: {
            show: false,
            tools: { 
              pan: true
            }
          },
          events: {
            mounted: function(ctx, config) {
              const highest1 = ctx.getHighestValueInSeries(0);
              const highest2 = ctx.getHighestValueInSeries(1);

              ctx.addPointAnnotation({
                x:  new Date(ctx.w.globals.seriesX[0][ctx.w.globals.series[0].indexOf(highest1)]).getTime(),
            y: highest1,
                label: {
                  style: {
                    cssClass: 'd-none'
                  }
                },
                customSVG: {
                    SVG: '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="#1b55e2" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" class="feather feather-circle"><circle cx="12" cy="12" r="10"></circle></svg>',
                    cssClass: undefined,
                    offsetX: -8,
                    offsetY: 5
                }
              })

              ctx.addPointAnnotation({
                x: new Date(ctx.w.globals.seriesX[1][ctx.w.globals.series[1].indexOf(highest2)]).getTime(),
                y: highest2,
                label: {
                  style: {
                    cssClass: 'd-none'
                  }
                },
                customSVG: {
                    SVG: '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="#e7515a" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" class="feather feather-circle"><circle cx="12" cy="12" r="10"></circle></svg>',
                    cssClass: undefined,
                    offsetX: -8,
                    offsetY: 5
                }
              })
            }
          }
        },
        colors: ['blue', 'red', 'purple', 'yellow', 'green'],
        dataLabels: {
            enabled: false
        },
        markers: {
          discrete: [{
          seriesIndex: 0,
          dataPointIndex: 7,
          fillColor: '#000',
          strokeColor: '#000',
          size: 5
        }, {
          seriesIndex: 2,
          dataPointIndex: 11,
          fillColor: '#000',
          strokeColor: '#000',
          size: 4
        }]
        },
        // volume
        title: {
          text: 'Stock Trends',
          align: 'right',
          floating: false,
          // offsetX: -100,
          offsetY: -8,
          style: {
            fontSize: '25px',
          color:  '#bfc9d4'
          },
        },
        stroke: {
            show: true,
            curve: 'smooth',
            width: 2,
            lineCap: 'square'
        },
        series: [{
            name: 'Open',
            data:  _data.map(i => [Date.parse(i.Date), i.Open])
        },
                {
            name: 'Low',
            data:  _data.map(i => [Date.parse(i.Date), i.Low])
        },
                {
            name: 'High',
            data:  _data.map(i => [Date.parse(i.Date), i.High])
        },
                 {
            name: 'Close',
            data:  _data.map(i => [Date.parse(i.Date), i.Close])
        },
                {
            name: 'Adj. CLose',
            data:  _data.map(i => [Date.parse(i.Date), i.Adj_close])
        }],
        xaxis: {
          type: 'datetime',
          labels: {
            datetimeFormatter: {
              year: 'yyyy',
              month: 'MMM \'yy',
              day: 'dd MMM',
              hour: 'HH:mm'
            }
          },
          axisBorder: {
            show: false
          },
          axisTicks: {
            show: false
          },
          crosshairs: {
            show: true
          },
          labels: {
            offsetX: 0,
            offsetY: 5,
            style: {
                fontSize: '12px',
                fontFamily: 'Nunito, sans-serif',
                cssClass: 'apexcharts-yaxis-title',
            },
          }
        },
        yaxis: {
              forceNiceScale: true,
          labels: {
            formatter: function(value, index) {
              return '$' + (value.toFixed(2)) + 'K'
            },
            offsetX: -25,
            offsetY: 0,
            style: {
                fontSize: '12px',
                fontFamily: 'Nunito, sans-serif',
                cssClass: 'apexcharts-yaxis-title',
            },
          }
        },
        grid: {
          borderColor: '#191e3a',
          strokeDashArray: 5,
          xaxis: {
              lines: {
                  show: true
              }
          },   
          yaxis: {
              lines: {
                  show: false,
              }
          }
        },
          padding: {
            top: 0,
            right: 0,
            bottom: 0,
            left: -10
        },  
        legend: {
          position: 'top',
          horizontalAlign: 'right',
          offsetY: -50,
            // width: 10,
          fontSize: '14px',
          fontFamily: 'Nunito, sans-serif',
          floating: true,
          markers: {
            width: 8,
            height: 8,
            strokeWidth: 12,
            strokeColor: ['blue', 'red', 'purple', 'yellow', 'green'],
            fillColors: ['blue', 'red', 'purple', 'yellow', 'green'],
            radius: 6,
            onClick: undefined,
            offsetX: 0,
            offsetY: 0
          },    
          itemMargin: {
            horizontal: 10,
            vertical: 0
          }
        },
        tooltip: {
          theme: 'dark',
          marker: {
            show: true,
          }
        },
        fill: {
            type:"gradient",
            gradient: {
                type: "vertical",
                shadeIntensity: 1,
                inverseColors: !1,
                opacityFrom: .28,
                opacityTo: .05,
                stops: [45, 100]
            }
        },
        responsive: [{
          breakpoint: 575,
          options: {
            legend: {
                offsetY: -30,
            },
          },
        }]
      };
        
      
      console.time("trends");

      var chart1 = new ApexCharts(
        document.querySelector("#trends"),
        options1
      );

      chart1.render();
      
      console.timeEnd("trends");
});



fetch("static/assets/js/dashboard/pastpreds.json").then(resp => resp.json())
    .then(_data => {

      var options2 = {
        chart: {
          fontFamily: 'Nunito, sans-serif',
          height: 365,
          type: 'area',
          zoom: {
            type: 'x',
              enabled: true
          },
          dropShadow: {
            enabled: true,
            opacity: 0.3,
            blur: 5,
            left: -7,
            top: 22
          },
          toolbar: {
            show: false,
            tools: { 
              pan: true
            }
          },
          events: {
            mounted: function(ctx, config) {
              const highest1 = ctx.getHighestValueInSeries(0);
              const highest2 = ctx.getHighestValueInSeries(1);

              ctx.addPointAnnotation({
                x:  new Date(ctx.w.globals.seriesX[0][ctx.w.globals.series[0].indexOf(highest1)]).getTime(),
             y: highest1,
                label: {
                  style: {
                    cssClass: 'd-none'
                  }
                },
                customSVG: {
                    SVG: '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="#1b55e2" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" class="feather feather-circle"><circle cx="12" cy="12" r="10"></circle></svg>',
                    cssClass: undefined,
                    offsetX: -8,
                    offsetY: 5
                }
              })

              ctx.addPointAnnotation({
                x: new Date(ctx.w.globals.seriesX[1][ctx.w.globals.series[1].indexOf(highest2)]).getTime(),
                y: highest2,
                label: {
                  style: {
                    cssClass: 'd-none'
                  }
                },
                customSVG: {
                    SVG: '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="#e7515a" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" class="feather feather-circle"><circle cx="12" cy="12" r="10"></circle></svg>',
                    cssClass: undefined,
                    offsetX: -8,
                    offsetY: 5
                }
              })
            }
          }
        },
        colors: ['blue', 'red', 'purple', 'yellow', 'green'],
        dataLabels: {
            enabled: false
        },
        markers: {
          discrete: [{
          seriesIndex: 0,
          dataPointIndex: 7,
          fillColor: '#000',
          strokeColor: '#000',
          size: 5
        }, {
          seriesIndex: 2,
          dataPointIndex: 11,
          fillColor: '#000',
          strokeColor: '#000',
          size: 4
        }]
        },
        // volume
        title: {
          text: 'Model Accuracy',
          align: 'right',
          floating: false,
          // offsetX: -100,
          offsetY: -8,
          style: {  
            fontSize: '25px',
          color:  '#bfc9d4'
          },
        },
        stroke: {
            show: true,
            curve: 'smooth',
            width: 2,
            lineCap: 'square'
        },
        series: [{
            name: 'Adj. Close',
            data:  _data.map(i => [Date.parse(i.Date), i.Adj_close])
                        },
                {
            name: 'ARIMA',
            data:  _data.map(i => [Date.parse(i.Date), i.ARIMA])
        }],
        xaxis: {
          type: 'datetime',
          labels: {
            datetimeFormatter: {
              year: 'yyyy',
              month: 'MMM \'yy',
              day: 'dd MMM',
              hour: 'HH:mm'
            }
          },
          axisBorder: {
            show: false
          },
          axisTicks: {
            show: false
          },
          crosshairs: {
            show: true
          },
          labels: {
            offsetX: 0,
            offsetY: 5,
            style: {
                fontSize: '12px',
                fontFamily: 'Nunito, sans-serif',
                cssClass: 'apexcharts-yaxis-title',
            },
          }
        },
        yaxis: {
            forceNiceScale: true,
          labels: {
            formatter: function(value, index) {
              return '$' + (value.toFixed(2)) + 'K'
            },
            offsetX: -25,
            offsetY: 0,
            style: {
                fontSize: '12px',
                fontFamily: 'Nunito, sans-serif',
                cssClass: 'apexcharts-yaxis-title',
            },
          }
        },
        grid: {
          borderColor: '#191e3a',
          strokeDashArray: 5,
          xaxis: {
              lines: {
                  show: true
              }
          },   
          yaxis: {
              lines: {
                  show: false,
              }
          }
        }, 
        legend: {
          position: 'top',
            // width: 4,
          horizontalAlign: 'right',
          offsetY: -50,
          floating: true,
          fontSize: '16px',
          fontFamily: 'Nunito, sans-serif',
          markers: {
            width: 8,
            height: 8,
            strokeWidth: 2,
            strokeColor: ['blue', 'red', 'purple', 'yellow', 'green'],
            fillColors: ['blue', 'red', 'purple', 'yellow', 'green'],
            radius: 12,
            onClick: undefined,
            offsetX: 0,
            offsetY: 0
          },    
          itemMargin: {
            horizontal: 10,
            vertical: 0
          }
        },
        tooltip: {
          theme: 'dark',
          marker: {
            show: true,
          }
        },
        fill: {
            type:"gradient",
            gradient: {
                type: "vertical",
                shadeIntensity: 1,
                inverseColors: !1,
                opacityFrom: .28,
                opacityTo: .05,
                stops: [45, 100]
            }
        },
        responsive: [{
          breakpoint: 575,
          options: {
            legend: {
                offsetY: -30,
            },
          },
        }]
      };
        
      
      console.time("preds");

      var chart2 = new ApexCharts(
        document.querySelector("#preds"),
        options2
      );

      chart2.render();
      
      console.timeEnd("preds");
});

fetch("static/assets/js/dashboard/forecast.json").then(resp => resp.json())
    .then(_data => {

      var options3 = {
        chart: {
          fontFamily: 'Nunito, sans-serif',
          height: 365,
          type: 'area',
          zoom: {
            type: 'x',
              enabled: true
          },
          dropShadow: {
            enabled: true,
            opacity: 0.3,
            blur: 5,
            left: -7,
            top: 22
          },
          toolbar: {
            show: false,
            tools: { 
              pan: true
            }
          },
          events: {
            mounted: function(ctx, config) {
              const highest1 = ctx.getHighestValueInSeries(0);
              const highest2 = ctx.getHighestValueInSeries(1);

              ctx.addPointAnnotation({
                x: new Date(ctx.w.globals.seriesX[0][ctx.w.globals.series[0].indexOf(highest1)]).getTime(),
                  y: highest1,
                label: {
                  style: {
                    cssClass: 'd-none'
                  }
                },
                customSVG: {
                    SVG: '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="#1b55e2" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" class="feather feather-circle"><circle cx="12" cy="12" r="10"></circle></svg>',
                    cssClass: undefined,
                    offsetX: -8,
                    offsetY: 5
                }
              })

              ctx.addPointAnnotation({
                x: new Date(ctx.w.globals.seriesX[1][ctx.w.globals.series[1].indexOf(highest2)]).getTime(),
                y: highest2,
                label: {
                  style: {
                    cssClass: 'd-none'
                  }
                },
                customSVG: {
                    SVG: '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="#e7515a" stroke="#fff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" class="feather feather-circle"><circle cx="12" cy="12" r="10"></circle></svg>',
                    cssClass: undefined,
                    offsetX: -8,
                    offsetY: 5
                }
              })
            }
          }
        },
        colors: ['blue', 'red', 'purple', 'yellow', 'green'],
        dataLabels: {
            enabled: false
        },
        markers: {
          discrete: [{
          seriesIndex: 0,
          dataPointIndex: 7,
          fillColor: '#000',
          strokeColor: '#000',
          size: 5
        }, {
          seriesIndex: 2,
          dataPointIndex: 11,
          fillColor: '#000',
          strokeColor: '#000',
          size: 4
        }]
        },
        // volume
        title: {
          text: '7 Days Forecast',
          align: 'right',
          floating: false,
          // offsetX: -100,
          offsetY: -5,
          style: {
            fontSize: '25px',
          color:  '#bfc9d4'
          },
        },
        stroke: {
            show: true,
            curve: 'smooth',
            width: 2,
            lineCap: 'square'
        },
        series: [{
            name: 'ARIMA',
            data:  _data.map(i => [Date.parse(i.Date), i.ARIMA])
        }],
        xaxis: {
          type: 'datetime',
          labels: {
            datetimeFormatter: {
              year: 'yyyy',
              month: 'MMM \'yy',
              day: 'dd MMM',
              hour: 'HH:mm'
            }
          },
          axisBorder: {
            show: false
          },
          axisTicks: {
            show: false
          },
          crosshairs: {
            show: true
          },
          labels: {
            offsetX: 0,
            offsetY: 5,
            style: {
                fontSize: '12px',
                fontFamily: 'Nunito, sans-serif',
                cssClass: 'apexcharts-yaxis-title',
            },
          }
        },
        yaxis: {
            forceNiceScale: true,
          labels: {
            formatter: function(value, index) {
              return '$' + (value.toFixed(2)) + 'K'
            },
            offsetX: -25,
            offsetY: 0,
            style: {
                fontSize: '12px',
                fontFamily: 'Nunito, sans-serif',
                cssClass: 'apexcharts-yaxis-title',
            },
          }
        },
        grid: {
          borderColor: '#191e3a',
          strokeDashArray: 5,
          xaxis: {
              lines: {
                  show: true
              }
          },   
          yaxis: {
              lines: {
                  show: false,
              }
          }
        }, 
        legend: {
          position: 'top',
          horizontalAlign: 'right',
          // width: 4,
          offsetY: -50,
          fontSize: '16px',
          floating: true,
          fontFamily: 'Nunito, sans-serif',
          markers: {
            width: 8,
            height: 8,
            strokeWidth: 2,
            strokeColor: ['blue', 'red', 'purple', 'yellow', 'green'],
            fillColors: ['blue', 'red', 'purple', 'yellow', 'green'],
            radius: 12,
            onClick: undefined,
            offsetX: 0,
            offsetY: 0
          },    
          itemMargin: {
            horizontal: 5,
            vertical: 0
          }
        },
        tooltip: {
          theme: 'dark',
          marker: {
            show: true,
          }
        },
        fill: {
            type:"gradient",
            gradient: {
                type: "vertical",
                shadeIntensity: 1,
                inverseColors: !1,
                opacityFrom: .28,
                opacityTo: .05,
                stops: [45, 100]
            }
        },
        responsive: [{
          breakpoint: 575,
          options: {
            legend: {
                offsetY: -30,
            },
          },
        }]
      };
        
      
      console.time("forecast");

      var chart3 = new ApexCharts(
        document.querySelector("#forecast"),
        options3
      );

      chart3.render();
      
      console.timeEnd("forecast");
});

/*
    =============================================
        Perfect Scrollbar | Recent Activities
    =============================================
*/
const ps = new PerfectScrollbar(document.querySelector('.mt-container'));


} catch(e) {
    console.log(e);
}