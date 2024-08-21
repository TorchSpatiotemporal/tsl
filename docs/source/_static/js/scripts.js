document.addEventListener("DOMContentLoaded", function (event) {

    async function fetchStars(repo) {
        const response = await fetch(`https://api.github.com/repos/${repo}`);
        return await response.json().stargazers_count;
    }

    async function fetchContributors(repo) {
        const response = await fetch(`https://api.github.com/repos/${repo}/contributors`);
        return await response.json();
    }

    const starsElems = document.querySelectorAll('.gh-star-counts');
    starsElems.forEach(function (elem) {
        fetchStars(elem.dataset.repo).then(stars => {
            elem.textContent = stars;
        });
    });

    const contribElems = document.querySelectorAll('.gh-contributors');
    contribElems.forEach(function (elem) {
        fetchContributors(elem.dataset.repo).then(contributors => {
            // contributors is a list of contributor objects
            // for each contributor, create a custom div element with class 'contributor' and append it to the elem
            // then populate the div with an img element with src=avatar_url, alt=login, title=login
            // add also a text node with the number of contributions
            // finally, add a link to the contributor's GitHub profile
            contributors.forEach(contributor => {
                const div = document.createElement('div');
                div.classList.add('gh-contributor');
                const a = document.createElement('a');
                div.appendChild(a);
                a.href = contributor.html_url;
                a.innerHTML = `<img src="${contributor.avatar_url}" alt="${contributor.login}" title="${contributor.login}">`;
                if (elem.hasAttribute("show-contributions-number")) {
                    a.innerHTML += `<span class="gh-contributor-number">${contributor.contributions}</span>`;
                }
                div.appendChild(a);
                elem.appendChild(div);
            });
        });
    });

});
