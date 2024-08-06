CREATE DATABASE curso_sql;

DROP DATABASE curso_sql;
DROP TABLE frameworks;
USE curso_sql;
SHOW TABLES;
SELECT * FROM armaduras_myisam;

# MOTORES DE TABLAS, solo se uso si la version es menor a 7 en MySQL
# MYISAM TABLA QUE NO SIRVE PARA MODELOS RELACIONALES COMPLEJOS 
CREATE TABLE armaduras_myisam(
	armadura_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    armadura VARCHAR(30) NOT NULL
) ENGINE = MyISAM DEFAULT CHARSET = utf8mb4; # para crear en formato MyIsam < 8 de la version

# Se podria especificar InnoDB en servidores, pero ya viene por default. Buscar engine tables
CREATE TABLE armaduras_innodb (
  armadura_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  armadura VARCHAR(30) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4; # Juego de caracteres en base de datos CHARSET


# CUANDO YA TIENES UN MODELO RELACIONAL, DIFICILMENTE SE PUEDE ELIMINAR TABLAS RELACIONADAS
# PARA ESO SE APLICA RESTRICCIONES TIPO DELETE AND UPDATE Y ESTAS SON:

/*
RESTRICCIONES (DELETE Y UPDATE)
  - CASCADE :  Si se elimina o actualiza una fila referenciada, también se eliminan o actualizan todas las filas relacionadas.
  - SET NULL  : Si se elimina o actualiza una fila referenciada, se establece el valor de la clave externa a NULL en las filas relacionadas.
  - SET DEFAULT : Si se elimina o actualiza una fila referenciada, se establece el valor de la clave externa al valor predeterminado especificado en las filas relacionadas.
  - RESTRICT : Impide la eliminación o actualización de una fila referenciada si existen filas relacionadas. Es el comportamiento predeterminado.
*/

DROP TABLE lenguajes;
DROP TABLE entornos;
TRUNCATE TABLE lenguajes;
#1. CREAMOS TABLA LENGUAJE
CREATE TABLE lenguajes(
	lenguaje_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    lenguaje VARCHAR(30) NOT NULL
);

#2. insertamos los registros de lenguaje
INSERT INTO lenguajes (lenguaje) VALUES
	("JavaScript"),
	("PHP"),
	("Python"),
	("Ruby"),
	("JAVA");
        
#3. creamos tabla de framewokrs
CREATE TABLE frameworks (
  framework_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  framework VARCHAR(30) NOT NULL,
  lenguaje INT UNSIGNED,
  FOREIGN KEY (lenguaje) REFERENCES lenguajes(lenguaje_id)
);


#4. insertamos registros de frameworks
INSERT INTO frameworks (framework, lenguaje) VALUES
  ("React", 1),
  ("Angular", 1),
  ("Vue", 1),
  ("Svelte", 1),
  ("Laravel", 2),
  ("Symfony", 2),
  ("Flask", 3),
  ("Django", 3),
  ("On Rails", 4);
  
SELECT * FROM lenguajes;
SELECT * FROM entornos;
SELECT * FROM frameworks;

#5. hacemos una vista interncepcion entre frameworks y lenguaje
SELECT *
	FROM frameworks f
	INNER JOIN lenguajes l
    ON f.lenguaje = l.lenguaje_id;
    # 7. vemos que no existe java por ningun sitio, ya que no tiene relacion solo ese dato se podria eliminar
    
#6. intentamos eliminar pyhton del lenguaje_id
DELETE FROM lenguajes WHERE lenguaje_id =3; # no nos permite eliminar por que ese campo tiene dependencias con la tabla de los frameworks
DELETE FROM lenguajes WHERE lenguaje_id =5; # 8. Java es el unico registro que se puede eliminar ya que no tiene relacion con ninguna.
DELETE FROM lenguajes WHERE lenguaje_id= 1; #9. Intentamos con javascrip, y vemos que no se puede eliminar ya que tiene FK

#10. Ahora probamos cambiando el ID, ojo no es buena practica
UPDATE lenguajes SET lenguaje_id = 13 WHERE lenguaje_id = 3; # no se puede actulizar un registro de la llave foreanea ya que tiene relacion

#11. ¿Entonces como se puede eliminar o actualizar un registro ya relacionado? Para eso se establece a la hora de crear tablas
#12. Eliminamos las tablas de lenguajes y frameworks

DROP TABLE lenguajes;
DROP TABLE frameworks;

#13. Agregamos la tabla lenguajes que sirve como EC y sus registros
CREATE TABLE lenguajes(
	lenguaje_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    lenguaje VARCHAR(30) NOT NULL
);

INSERT INTO lenguajes (lenguaje) VALUES
	("JavaScript"),
	("PHP"),
	("Python"),
	("Ruby"),
	("JAVA");

#14. donde se genera las restricciones es donde hay llaves foraneas, entocnes es ahi donde se tiene que agregar restricciones
CREATE TABLE frameworks (
  framework_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  framework VARCHAR(30) NOT NULL,
  lenguaje INT UNSIGNED,
  FOREIGN KEY (lenguaje) REFERENCES lenguajes(lenguaje_id)
    ON DELETE RESTRICT ON UPDATE CASCADE #15.Cuando pretenda eliminar un registro restringelo, en cambio cuando pretendan actualizar realiza el cambio en cascada
#ON DELETE SET NULL ON UPDATE CASCADE
);

INSERT INTO frameworks (framework, lenguaje) VALUES
  ("React", 1),
  ("Angular", 1),
  ("Vue", 1),
  ("Svelte", 1),
  ("Laravel", 2),
  ("Symfony", 2),
  ("Flask", 3),
  ("Django", 3),
  ("On Rails", 4);
  
  SELECT * FROM lenguajes;
  SELECT * FROM frameworks;
  
  # 16. repetimos los pasos anteriores con join
  SELECT *
	FROM frameworks f
	INNER JOIN lenguajes l
    ON f.lenguaje = l.lenguaje_id; #19. 
    
    #17. intentamos eliminar el id 3 nuevamente
    DELETE FROM lenguajes WHERE lenguaje_id =3; # 18.no se puede eliminar por que establecimos delete restric

#18.ahora intentamos actualizando
UPDATE lenguajes SET lenguaje_id = 13 WHERE lenguaje_id = 3; #19, ahora si se puede actualizar, ya que pusimos que se actulce en cascada

DROP TABLE lenguajes;
DROP TABLE frameworks;

# 20. restricciones multiples 
CREATE TABLE lenguajes(
	lenguaje_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    lenguaje VARCHAR(30) NOT NULL
);

INSERT INTO lenguajes (lenguaje) VALUES
	("JavaScript"),
	("PHP"),
	("Python"),
	("Ruby"),
	("JAVA");
    
CREATE TABLE entornos(
	entorno_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    entorno VARCHAR(30) NOT NULL
);

INSERT INTO entornos (entorno) VALUES
	("Frontend"),
	("Backend");
    
CREATE TABLE frameworks(
	framework_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    framework VARCHAR(30) NOT NULL,
    lenguaje INT UNSIGNED,
    entorno INT UNSIGNED,
    # se agrega restricciones por cada foranes, eliminar restring, update cascada
    FOREIGN KEY (lenguaje)
		REFERENCES lenguajes(lenguaje_id)
        ON DELETE RESTRICT
        ON UPDATE CASCADE,
	FOREIGN KEY (entorno)
		REFERENCES entornos(entorno_id)
        ON DELETE RESTRICT
        ON UPDATE CASCADE
);

INSERT INTO frameworks (framework, lenguaje, entorno) VALUES
  ("React", 1, 1),
  ("Angular", 1, 1),
  ("Vue", 1, 1),
  ("Svelte", 1, 1),
  ("Laravel", 2, 2),
  ("Symfony", 2, 2),
  ("Flask", 3, 2),
  ("Django", 3, 2),
  ("On Rails", 4, 2);
  
SELECT * FROM lenguajes;
  SELECT * FROM frameworks;
  SELECT * FROM entornos;
  
#21. realizamos un nuevo join pero agregando a entornos
SELECT *
	FROM frameworks f
	INNER JOIN lenguajes l ON f.lenguaje = l.lenguaje_id
    INNER JOIN entornos e ON f.entorno = e.entorno_id;

#22. INTENTAMOS ELIMINAR ENTORNOS
DELETE FROM entornos WHERE entorno_id = 1;#23. NO NOS DIEJA POR QUE TENEMOS EL RESTRIC
UPDATE entornos SET entorno_id = 19 WHERE entorno_id = 1; #24. EN CAMBIO SI NOS DEJA ACTUALIZAR

---
# TRANSACCIONES, ES UN CONJUNTO DE INSTRUCCIONES SQL QUE SE REALIZA UNO DE TRAS DE OTRA
DELETE FROM frameworks;
SELECT * FROM frameworks;
# 1. una transaccion inicia con 
START TRANSACTION; # se incia con esto cntrl + enter
	UPDATE frameworks SET framework = "Vue.js" WHERE framework_id = 2; # segundo esto
    DELETE FROM frameworks;
    INSERT INTO frameworks VALUES (0,"Spring", 5,2);
    
ROLLBACK; # en caso ocurra un error se regresa 
COMMIT; # en caso estes satisfecho con el resultado se envia

# LIMIT
SELECT * FROM frameworks;
SELECT * FROM frameworks LIMIT 2;
SELECT * FROM frameworks LIMIT 2,2;
SELECT * FROM frameworks LIMIT 4, 2;
SELECT * FROM frameworks LIMIT 6, 2;
SELECT * FROM frameworks LIMIT 8, 2;
SELECT * FROM frameworks LIMIT 10, 2;
SELECT * FROM frameworks LIMIT 8, 2;

# ENCRIPTAR INFORMACION como buenas practicas

SELECT MD5('m1 Sup3r P4$$w0rD'); #MD5 convierte una cadena de textos a valores tipo hash de 128bits
SELECT SHA1('m1 Sup3r P4$$w0rD'); #SHA1 Mas segurom y genera un hash de 160bits
SELECT SHA2('m1 Sup3r P4$$w0rD', 256);# SHA2, TE INVITA UN # DE BITS, A MAS BITS MAS DIFICIL DE HACKEAR

SELECT AES_ENCRYPT('m1 Sup3r P4$$w0rD', 'llave_secreta'); # FACTOR DE DOBLE AUTOENTICACION, MAS SEGURO
SELECT AES_DECRYPT(nombre_campo, 'llave_secreta'); # PARA DESENCRIPTAR 

# EJEMPLO DE ALMACENADO DE 16 DIGITOS DE UNA TARJETA DE CRREDITO

CREATE TABLE pagos_recurrentes(
  cuenta VARCHAR(8) PRIMARY KEY,
  nombre VARCHAR(50) NOT NULL,
  tarjeta BLOB # BLOB ALMACENAR DATOS ENCRIPTADOS O DATOS BINARIOS
);

INSERT INTO pagos_recurrentes VALUES
  ('12345678', 'Jon', AES_ENCRYPT('1234567890123488', '12345678')),
  ('12345677', 'Irma', AES_ENCRYPT('1234567890123477', '12345677')),
  ('12345676', 'Kenai', AES_ENCRYPT('1234567890123466', '12345676')),
  ('12345674', 'Kala', AES_ENCRYPT('1234567890123455', 'super_llave')),
  ('12345673', 'Miguel', AES_ENCRYPT('1234567890123444', 'super_llave'));

SELECT * FROM pagos_recurrentes;

# PARA DESENCRIPTAR CASTEO O CASTING CON CAST Y PASAMOS A CHAR
SELECT CAST(AES_DECRYPT(tarjeta, '12345678') AS CHAR) AS tdc, nombre
  FROM pagos_recurrentes;  

SELECT CAST(AES_DECRYPT(tarjeta, 'super_llave') AS CHAR) AS tdc, nombre
  FROM pagos_recurrentes;  

SELECT CAST(AES_DECRYPT(tarjeta, 'qwerty') AS CHAR) AS tdc, nombre
  FROM pagos_recurrentes;  