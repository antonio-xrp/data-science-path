CREATE DATABASE curso_sql;
DROP DATABASE curso_sql;
USE curso_sql;
SHOW TABLES;
DROP TABLE caballeros;
TRUNCATE TABLE caballeros;
SHOW INDEX FROM caballeros;
# Indices son como los libros, nos ayudan a ser mas rapido para las busquedas, hay 3 tipos de indices, primary key, UQ, y las indices que puedas crear

#1. Definiendo nuestros indices
CREATE TABLE caballeros (
caballero_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
nombre VARCHAR(30),
armadura VARCHAR(30) UNIQUE,
  rango VARCHAR(30),
  signo VARCHAR(30),
  ejercito VARCHAR(30),
  pais VARCHAR(30),
  #definir indices que no son llave primaria ni campo unico
  INDEX i_rango (rango),
  INDEX i_signo (signo),
  #multiple indice
  INDEX i_caballeros (ejercito,pais)
);

#2. Indice tipo busqueda google, full text a partir de varios campos, demanda mas tiempo al motor de base de datos
CREATE TABLE caballeros(
caballero_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
nombre VARCHAR(30),
  armadura VARCHAR(30),
  rango VARCHAR(30),
  signo VARCHAR(30),
  ejercito VARCHAR(30),
  pais VARCHAR(30),
  # campo de fulltext "fi_search" 
  FULLTEXT INDEX fi_search (armadura, rango, signo, ejercito, pais)
);

# 3. Sin indices
CREATE TABLE caballeros (
  caballero_id INT UNSIGNED,
  nombre VARCHAR(30),
  armadura VARCHAR(30),
  rango VARCHAR(30),
  signo VARCHAR(30),
  ejercito VARCHAR(30),
  pais VARCHAR(30)
);

INSERT INTO caballeros VALUES
	(0, "Seiya","Pegaso","Bronce","Sagitario","Athena","Japón"),
	(0,"Shiryu","Dragón","Bronce","Libra","Athena","Japón"),
  (0,"Hyoga","Cisne","Bronce","Acuario","Athena","Rusia"),
  (0,"Shun","Andromeda","Bronce","Virgo","Athena","Japón"),
  (0,"Ikki","Fénix","Bronce","Leo","Athena","Japón"),
  (0,"Kanon","Géminis","Oro","Géminis","Athena","Grecia"),
  (0,"Saga","Junini","Oro","Junini","Athena","Grecia"),
  (0,"Camus","Acuario","Oro","Acuario","Athena","Francia"),
  (0,"Rhadamanthys","Wyvern","Espectro","Escorpión Oro","Hades","Inglaterra"),
  (0,"Kanon","Dragón Marino","Marino","Géminis Oro","Poseidón","Grecia"),
  (0,"Kagaho","Bennu","Espectro","Leo","Hades","Rusia");
  
SELECT * FROM caballeros;

SELECT * FROM caballeros WHERE signo = "Leo";

#2. full text sintaxis , match para hacer coincidencia, lo cual nos dice busca en estas columnas
SELECT * FROM caballeros
  WHERE MATCH(armadura, rango, signo, ejercito, pais)
  AGAINST('Oro' IN BOOLEAN MODE);
  
SHOW INDEX FROM caballeros;


#3. para agregar indices a una tabla existente si n la necesidad de eliminar
ALTER TABLE caballeros ADD CONSTRAINT pk_caballero_id PRIMARY KEY (caballero_id);

# PARA HACER AUTO INCREMENTAL
ALTER TABLE caballeros MODIFY COLUMN caballero_id INT AUTO_INCREMENT;
ALTER TABLE caballeros ADD CONSTRAINT uq_armadura UNIQUE (armadura);

# para eliminar un index
ALTER TABLE caballeros DROP CONSTRAINT uq_armadura;

#para agregar otros indices que no sean PK UQ
ALTER TABLE caballeros ADD INDEX i_rango (rango);
ALTER TABLE caballeros DROP INDEX i_rango;

# para agregar multiple indice
ALTER TABLE caballeros ADD INDEX i_ejercito_pais (ejercito, pais);
ALTER TABLE caballeros DROP INDEX i_ejercito_pais;

#PARA AGREGAR FULL TEXT INDICE
ALTER TABLE caballeros ADD FULLTEXT INDEX fi_search (nombre,signo);
ALTER TABLE caballeros DROP INDEX fi_search;