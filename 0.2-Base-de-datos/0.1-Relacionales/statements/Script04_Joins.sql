CREATE DATABASE curso_sql;

DROP DATABASE curso_sql;
USE curso_sql;
DROP TABLE caballeros;
TRUNCATE TABLE caballeros;

SELECT * FROM caballeros;
SELECT * FROM armaduras;
SELECT * FROM rangos;
SELECT * FROM signos;
SELECT * FROM ejercitos;
SHOW TABLES;

CREATE TABLE armaduras (
armadura_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
armadura VARCHAR(30) NOT NULL
);

CREATE TABLE signos (
  signo_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  signo VARCHAR(30) NOT NULL
);

CREATE TABLE rangos (
  rango_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  rango VARCHAR(30) NOT NULL
);

CREATE TABLE ejercitos (
  ejercito_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  ejercito VARCHAR(30) NOT NULL
);

CREATE TABLE paises (
  pais_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  pais VARCHAR(30) NOT NULL
);

CREATE TABLE caballeros(
	caballero_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(30),
    armadura INT UNSIGNED,
    rango INT UNSIGNED,
    signo INT UNSIGNED,
    ejercito INT UNSIGNED,
    pais INT UNSIGNED,
    FOREIGN KEY(armadura) REFERENCES armaduras(armadura_id),
    FOREIGN KEY(rango) REFERENCES rangos(rango_id),
    FOREIGN KEY(signo) REFERENCES signos(signo_id),
    FOREIGN KEY(ejercito) REFERENCES ejercitos(ejercito_id),
    FOREIGN KEY(pais) REFERENCES paises(pais_id)
);

INSERT INTO armaduras values
(1,"Pegaso"),
  (2, "Dragón"),
  (3, "Cisne"),
  (4, "Andrómeda"),
  (5, "Fénix"),
  (6, "Géminis"),
  (7, "Acuario"),
  (8, "Wyvern"),
  (9, "Dragón Marino"),
  (10, "Bennu");
  
  INSERT INTO rangos values
  (1, "Bronce"),
  (2, "Oro"),
  (3, "Espectro"),
  (4, "Marino");
  
  INSERT INTO signos VALUES
  (1, "Aries"),
  (2, "Tauro"),
  (3, "Géminis"),
  (4, "Cancer"),
  (5, "Leo"),
  (6, "Virgo"),
  (7, "Libra"),
  (8, "Escorpión"),
  (9, "Sagitario"),
  (10, "Capricornio"),
  (11, "Acuario"),
  (12, "Piscis");
  
  INSERT INTO ejercitos VALUES
    (1, "Athena"),
  (2, "Hades"),
  (3, "Poseidón");
  
  INSERT INTO paises VALUES
   (1, "Japón"),
  (2, "Rusia"),
  (3, "Grecia"),
  (4, "Francia"),
  (5, "Inglaterra");
  
INSERT INTO caballeros VALUES
  (1,"Seiya", 1, 1, 9, 1, 1),
  (2,"Shiryu", 2, 1, 7, 1, 1),
  (3,"Hyoga", 3, 1, 11, 1, 2),
  (4,"Shun", 4, 1, 6, 1, 1),
  (5,"Ikki", 5, 1, 5, 1, 1),
  (6,"Kanon", 6, 2, 3, 1, 3),
  (7,"Saga", 6, 2, 3, 1, 3),
  (8,"Camus", 7, 2, 11, 1, 4),
  (9,"Rhadamanthys", 8, 3, 8, 2, 5),
  (10,"Kanon", 9, 4, 3, 3, 3),
  (11,"Kagaho", 10, 3, 5, 2, 2);
  
SELECT * FROM armaduras;
SELECT * FROM rangos;
SELECT * FROM signos;
SELECT * FROM ejercitos;
SELECT * FROM paises;
SELECT * FROM caballeros;

# JOIN, caballeros es izq
# 1. Left Join:  C ES COMO UN ALIAS
SELECT * FROM caballeros C
	LEFT JOIN signos s
    ON c.signo = s.signo_id;

# right join
SELECT * FROM caballeros C
	RIGHT JOIN signos s
    ON c.signo = s.signo_id;

 # INNER JOIN   
SELECT * FROM caballeros c
	INNER JOIN signos s
    ON c.signo = s.signo_id;
 
 # full join
SELECT * FROM caballeros c
	LEFT JOIN signos s
    ON c.signo = s.signo_id
UNION
SELECT * FROM caballeros c
	RIGHT JOIN signos s
    ON c.signo = s.signo_id;
    

# join multiple
SELECT c.caballero_id, c.nombre, a.armadura,
	s.signo, r.rango, e.ejercito, p.pais
    FROM caballeros c
    INNER JOIN armaduras a ON c.armadura = a.armadura_id
    INNER JOIN signos s ON c.signo = s.signo_id
    INNER JOIN rangos r ON c.rango = r.rango_id
    INNER JOIN ejercitos e ON c.ejercito = e.ejercito_id
    INNER JOIN paises p ON c.pais = p.pais_id;
  
# sub consulta, una consulta dentro de otra
#1. contar el total de caballeros por signos
SELECT signo,
	(SELECT COUNT(*) FROM caballeros c WHERE c.signo = s.signo_id)
	AS total_caballeros
    FROM signos s;

#2. contar el total de caballeros por rango
SELECT rango,
	(SELECT COUNT(*) FROM caballeros c WHERE c.rango = r.rango_id)
    AS total_caballeros
    FROM rangos r;

#3. contar el total de caballeros por ejercito
SELECT ejercito,
	(SELECT COUNT(*) FROM CABALLEROS c WHERE c.ejercito = e.ejercito_id)
    AS total_caballeros
    FROM ejercitos e;

#4 contar el total de caballeros por paises
SELECT pais,
	(SELECT COUNT(*) FROM CABALLEROS c WHERE c.pais = p.pais_id)
    AS total_caballeros
    FROM paises p;
  
 # VISTA UNA TABLA VIRTUAL QUE SE DERIVA DE UNA O DE VARIAS EXISTENTE DENTRO DE VARIAS, CONSULTA PREDEFINA Y PRE EXISTENTE DENTRO DE LA BASE DE DATOS
# lAS VISTAS SON FILTROS DENTRO DE LA LOGICA DE NEGOCIOS YA ESTABLECIDOS Y AL FINAL PODRIA CONSIDERASE COMO UN ATAJO
# NORMALMENTE LOS REPORTES SEMANAS, MENSUALES, O ALGO ESPECIFICO SE GUARDA EN VISTAS 
CREATE VIEW vista_caballeros AS 
	SELECT c.caballero_id, c.nombre, a.armadura,caballeroscaballeros
		s.signo, r.rango, e.ejercito, p.pais
		FROM caballeros c
		INNER JOIN armaduras a ON c.armadura = a.armadura_id
		INNER JOIN signos s ON c.signo = s.signo_id
		INNER JOIN rangos r ON c.rango = r.rango_id
		INNER JOIN ejercitos e ON c.ejercito = e.ejercito_id
		INNER JOIN paises p ON c.pais = p.pais_id;
        
CREATE VIEW vista_signos AS 
	SELECT signo,
		(SELECT COUNT(*) FROM caballeros C WHERE c.signo = s.signo_id)
        AS total_caballeros
        FROM signos s;
        
# SE EJECUTA LAS VISTAS COMO SI FUERAN UN TABLA
SELECT * FROM vista_caballeros;
SELECT * FROM vista_signos;

# para eliminar una vista
DROP VIEW vista_caballeros;

# PARA VER LAS VISTAS CREADAS
SHOW FULL TABLES IN curso_sql WHERE TABLE_TYPE LIKE 'VIEW';