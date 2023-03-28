particlesJS("particles-js", {
    "particles": {
        "number": {
            "value": 150,
            "density": {
                "enable": true,
                "value_area": 500
            }
        },
        "color": {
            "value": "#ee4c2c"
        },
        "shape": {
            "type": "circle",
            "stroke": {
                "width": 0,
            }
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
            "opacity": 0.4,
            "width": 2
        },
        "move": {
            "enable": true,
            "speed": 1,
            "direction": "none",
            "random": true,
            "straight": false,
            "out_mode": "out",
            "bounce": false,
            "attract": {
                "enable": false,
                "rotateX": 600,
                "rotateY": 1200
            }
        }
    },
    "interactivity": {
        "detect_on": "canvas",
        "events": {
            "onhover": {
                "enable": true,
                "mode": "repulse"
            },
            "onclick": {
                "enable": false,
                "mode": "repulse"
            },
            "resize": true
        },
        "modes": {
            "repulse": {
                "distance": 30,
                "duration": 0.4
            }
        }
    },
    "retina_detect": true
});

// document.querySelector("#particles-js > .particles-js-canvas-el").height = 420;
// window.dispatchEvent(new Event('resize'));
