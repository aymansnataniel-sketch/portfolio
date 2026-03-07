-- Pregunta 1
-- ¿Quién es el actor con más series?
SELECT actor_id, COUNT(serie_id) AS actuacion 
FROM actuaciones 
GROUP BY actor_id 
ORDER BY actuacion DESC 
LIMIT 1;

-- Pregunta 2
-- ¿Cuál es la serie con mejor rating?
SELECT serie_id, AVG(rating_imdb) AS rating 
FROM episodios
GROUP BY serie_id
ORDER BY rating DESC
LIMIT 1;

-- Pregunta 3
-- ¿Cuál es el episodio con la duración más larga?
SELECT episodio_id, duracion 
FROM episodios
ORDER BY duracion DESC
LIMIT 1;
