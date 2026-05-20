document.addEventListener('DOMContentLoaded', () => {
    const searchInput = document.getElementById('searchInput');
    const searchBtn = document.getElementById('searchBtn');
    const autocompleteResults = document.getElementById('autocompleteResults');
    const resultsSection = document.getElementById('resultsSection');
    const loadingSection = document.getElementById('loadingSection');
    const errorSection = document.getElementById('errorSection');
    const statsContainer = document.getElementById('statsContainer');
    const sortRatingToggle = document.getElementById('sortRatingToggle');
    const modelSelect = document.getElementById('modelSelect');

    let debounceTimer;

    // Load Stats on mount
    fetch('/api/stats')
        .then(res => res.json())
        .then(data => {
            statsContainer.innerHTML = `
                <div class="stat-box">
                    <h3>${data.total_anime.toLocaleString()}</h3>
                    <p>Total Anime</p>
                </div>
                <div class="stat-box">
                    <h3>${data.avg_rating}</h3>
                    <p>Avg Rating</p>
                </div>
            `;
        })
        .catch(err => console.error("Could not load stats", err));

    // Search Autocomplete
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.trim();
        
        clearTimeout(debounceTimer);
        
        if (query.length < 2) {
            autocompleteResults.classList.add('hidden');
            return;
        }

        debounceTimer = setTimeout(() => {
            fetch(`/api/search?q=${encodeURIComponent(query)}&limit=5`)
                .then(res => res.json())
                .then(data => {
                    autocompleteResults.innerHTML = '';
                    if (data.length > 0) {
                        data.forEach(anime => {
                            const item = document.createElement('div');
                            item.className = 'autocomplete-item';
                            item.innerHTML = `
                                <span class="title">${anime.name}</span>
                                <span class="badge">${anime.type || 'N/A'}</span>
                            `;
                            item.addEventListener('click', () => {
                                searchInput.value = anime.name;
                                autocompleteResults.classList.add('hidden');
                                fetchRecommendations(anime.name);
                            });
                            autocompleteResults.appendChild(item);
                        });
                        autocompleteResults.classList.remove('hidden');
                    } else {
                        autocompleteResults.classList.add('hidden');
                    }
                });
        }, 300);
    });

    // Hide autocomplete when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-box')) {
            autocompleteResults.classList.add('hidden');
        }
    });

    // Handle Search Button Click
    searchBtn.addEventListener('click', () => {
        const query = searchInput.value.trim();
        if (query) {
            fetchRecommendations(query);
        }
    });

    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const query = searchInput.value.trim();
            if (query) {
                autocompleteResults.classList.add('hidden');
                fetchRecommendations(query);
            }
        }
    });

    // Handle Model or Sort Toggle Changes (Auto-refresh)
    [modelSelect, sortRatingToggle].forEach(el => {
        el.addEventListener('change', () => {
            const query = searchInput.value.trim();
            if (query && !resultsSection.classList.contains('hidden')) {
                fetchRecommendations(query);
            }
        });
    });

    // Fetch Recommendations
    function fetchRecommendations(title) {
        // Show loading state
        resultsSection.classList.add('hidden');
        errorSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');

        const sortRating = sortRatingToggle.checked;
        const modelType = modelSelect.value;
        const url = `/api/recommend?title=${encodeURIComponent(title)}&model=${encodeURIComponent(modelType)}&limit=12&sort_by_rating=${sortRating}`;

        fetch(url)
            .then(res => res.json())
            .then(data => {
                loadingSection.classList.add('hidden');
                
                if (data.error) {
                    showError(data.error);
                    return;
                }

                if (data.length === 0) {
                    showError("No recommendations found for this anime.");
                    return;
                }

                renderResults(data);
            })
            .catch(err => {
                loadingSection.classList.add('hidden');
                showError("An error occurred while fetching recommendations.");
                console.error(err);
            });
    }

    // Render the cards
    function renderResults(data) {
        resultsSection.innerHTML = '';
        
        data.forEach(anime => {
            const tags = anime.genre ? anime.genre.split(',').map(g => `<span class="tag">${g.trim()}</span>`).join('') : '';
            
            const card = document.createElement('div');
            card.className = 'anime-card';
            
            card.innerHTML = `
                <div class="card-header">
                    <div class="card-title">${anime.name}</div>
                    <div class="rating-badge">
                        <i class="fas fa-star"></i> ${anime.rating || 'N/A'}
                    </div>
                </div>
                <div class="card-tags">
                    <span class="tag" style="border-color: var(--accent-color); color: var(--accent-color)">${anime.type || 'Unknown'}</span>
                    ${tags}
                </div>
                <div class="card-footer">
                    <span><i class="fas fa-users"></i> ${(anime.members || 0).toLocaleString()}</span>
                    <span class="similarity-score">${(anime.similarity * 100).toFixed(1)}% Match</span>
                </div>
            `;
            
            resultsSection.appendChild(card);
        });

        resultsSection.classList.remove('hidden');
        
        // Scroll to results smoothly
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function showError(message) {
        errorSection.textContent = message;
        errorSection.classList.remove('hidden');
    }
});
