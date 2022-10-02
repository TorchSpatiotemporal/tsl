particlesJS("particles-js", {
    "particles": {
        "number": {
            "value": 80,
            "density": {"enable": true, "value_area": 500}
        },
        "color": {"value": "#ee4c2c"},
        "shape": {
            "type": "circle",
            "stroke": {"width": 0},
        },
        "opacity": {
            "value": 1.,
            "random": false,
            "anim": {
                "enable": false,
                "speed": 1,
                "opacity_min": 0.1,
                "sync": false
            }
        },
        "size": {
            "value": 5,
            "random": true,
            "anim": {
                "enable": false,
                "speed": 40,
                "size_min": 0.1,
                "sync": false
            }
        },
        "line_linked": {
            "enable": true,
            "distance": 150,
            "color": "#ee4c2c",
            "opacity": 0.8,
            "width": 2
        },
        "move": {
            "enable": true,
            "speed": 3,
            "direction": "none",
            "random": false,
            "straight": false,
            "out_mode": "bounce",
            "bounce": true,
            "attract": {"enable": false, "rotateX": 600, "rotateY": 1200}
        }
    },
    "interactivity": {
        "detect_on": "canvas",
        "events": {
            "onhover": {"enable": false, "mode": "repulse"},
            "onclick": {"enable": false, "mode": "push"},
            "resize": true
        }
    },
    "retina_detect": true
});

document.querySelector("#particles-js > .particles-js-canvas-el").height = 420;
window.dispatchEvent(new Event('resize'));