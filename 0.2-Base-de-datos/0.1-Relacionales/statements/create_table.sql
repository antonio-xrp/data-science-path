CREATE DATABASE curso_sql;
CREATE DATABASE IF NOT EXISTS curso_sql;

# Para seleccionar una base de datos lo cual usaremos
USE antonio_db;

# PARA MOSTRAR LAS TABLAS DE UNA BASE DE DATOS
SHOW TABLES;

# PARA describir una tabla en particular `DESCRIBE nombre_tabla`
DESCRIBE host_sumary;

# PARA CREAR UNA TABLA
CREATE TABLE usuarios(
	nombre VARCHAR(50),
    correo VARCHAR(50)
);

# Para agregar una columna más a una tabla
ALTER TABLE usuarios ADD COLUMN cumpleaños VARCHAR(15);

# Para modificar el tipo de dato de una columna
ALTER TABLE usuarios MODIFY cumpleaños DATE;

# Para modificar el nombre de una columna
ALTER TABLE usuarios RENAME COLUMN cumpleaños to nacimiento;

# Para eliminar una columna de una tabla
ALTER TABLE usuarios DROP COLUMN nacimiento;

# Para eliminar una tabla
DROP TABLE usuarios;

# Creamos una tabla con una llave primaria usuario_id, 
# de modo INT autoincrementable y UNSIGNED que significa entero pero sin signo

CREATE TABLE usuarios(
	usuario_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(30) NOT NULL,
    apellidos VARCHAR(59) NOT NULL,
    correo VARCHAR(50) UNIQUE,
    direccion VARCHAR(100) DEFAULT "SIN DIRECCION",
    edad INT DEFAULT 0
);