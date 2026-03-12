-- ¿Qué géneros son más prevalentes en la base de datos NetflixDB? 
-- Genera una lista de los distintos géneros y la cantidad de series por cada uno.

SELECT genero, COUNT(serie_id) AS cantidad FROM series
GROUP BY genero;

-- ¿Cuáles son las tres series con mayor rating IMDB y cuántos episodios tienen? 

SELECT series.titulo AS titulo, AVG (rating_imdb) AS rating, COUNT(episodios.episodio_id) AS recuento FROM episodios
JOIN series
ON series.serie_id = episodios.serie_id
GROUP BY series.serie_id
ORDER BY rating DESC
LIMIT 3;

-- ¿Cuál es la duración total de todos los episodios para la serie "Stranger Things"? 

SELECT SUM(episodios.duracion) AS "duración" FROM episodios
JOIN series
ON series.serie_id = episodios.serie_id
WHERE series.titulo = "Stranger Things"
