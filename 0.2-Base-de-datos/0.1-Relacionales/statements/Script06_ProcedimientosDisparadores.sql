USE curso_sql;
SHOW TABLES;
SELECT * FROM suscripciones;
# stop procidious o procedimientos almacenados --> conjunto de instrucciones de SQL que se almacena en la base de datos como si fuera una funcion
# por lo tanto lo podemos llamar y ejecutar tantas veces como lo necesitamos
# se utiliza para encapsular la logica de negocio y reducir la complejidad las complicaciones a la hora de interactuar con la base de datos
CREATE TABLE suscripciones(
	suscripcion_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    suscripcion VARCHAR(30) NOT NULL,
    costo DECIMAL(5,2) NOT NULL
);

INSERT INTO suscripciones VALUES
  (0, 'Bronce', 199.99),
  (0, 'Plata', 299.99),
  (0, 'Oro', 399.99);
  
CREATE TABLE clientes (
  cliente_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  nombre VARCHAR(30) NOT NULL,
  correo VARCHAR(50) UNIQUE
);

CREATE TABLE tarjetas (
  tarjeta_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  cliente INT UNSIGNED,
  tarjeta BLOB,
  FOREIGN KEY (cliente)
    REFERENCES clientes(cliente_id)
    ON DELETE RESTRICT
    ON UPDATE CASCADE
);

CREATE TABLE servicios(
  servicio_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  cliente INT UNSIGNED,
  tarjeta INT UNSIGNED,
  suscripcion INT UNSIGNED,
  FOREIGN KEY(cliente)
    REFERENCES clientes(cliente_id)
    ON DELETE RESTRICT
    ON UPDATE CASCADE,
  FOREIGN KEY(tarjeta)
    REFERENCES tarjetas(tarjeta_id)
    ON DELETE RESTRICT
    ON UPDATE CASCADE,
  FOREIGN KEY(suscripcion)
    REFERENCES suscripciones(suscripcion_id)
    ON DELETE RESTRICT
    ON UPDATE CASCADE
);

SELECT * FROM suscripciones;
SELECT * FROM clientes;
SELECT * FROM tarjetas;
SELECT * FROM servicios;
SELECT * FROM actividad_clientes;

#SE USA DELIMITIR PARA CREAR EL STOP PROCEDURE
DELIMITER //
CREATE PROCEDURE sp_asignar_servicio(
	IN i_suscripcion INT UNSIGNED, #1. DEFINIMOS PARAMETROS DE ENTRADA CON LA FUNCION IN
    IN i_nombre VARCHAR(30),
    IN i_correo VARCHAR(50),
    IN i_tarjeta VARCHAR(16),
    OUT o_respuesta VARCHAR(50) #2. SE CREA UN PARAMETRO DE SALIDA
)
	BEGIN
		#5. VALIDAMOS LAS VARIABLES
		DECLARE existe_correo INT DEFAULT 0; #5. VALIDAMOS LA VARIABLE EXISTE CORREO
        DECLARE cliente_id INT DEFAULT 0; #12. VALIDAMOS LA VARIABLE CLIENTE_ID
        DECLARE tarjeta_id INT DEFAULT 0; #14. VALIDAMOS TARJETA_ID PARA OBTNER LA EL ULTIMO ID INSERTADO
        
        START TRANSACTION; #3. DEFINIMOS EL INICIO
			
            SELECT COUNT(*) INTO existe_correo #6. ALMACENAMOS ESA VARIABLE DENTRO DE EXISTE_CORREO
				FROM clientes #6. DE LA TABLA CLIENTES
                WHERE correo = i_correo; #7. VALIDAMOS 
                
			IF existe_correo <> 0 THEN #8. SI EXISTE TU CORREO QUE ES DIFERENTE A 0 ENTONCES
				
                SELECT 'Tu correo ya ha sido registrado' INTO o_respuesta; #9. PRINT RESPUESTA Y SE ALMACENA EN VARIABLE DE SALIDA
			
            ELSE #10. DE LO CONTRARIO... SI ES IGUAL A 0, SIGNIFICA QUE NO HA REGISTRADO SU CORREO
            
				INSERT INTO clientes VALUES (0, i_nombre, i_correo); # 11. INSERTAMOS CLIENTES,
                SELECT LAST_INSERT_ID() INTO cliente_id; #12 NOS DEVUELVE EL ULTIMO ID QUE FUE REGISTRADO EN LA TABLA QUE INDIQUEMOS
                
                INSERT INTO tarjetas #13. INSERTAMOS EL RESULTADO EN TARJETAS
					VALUES(0,cliente_id, AES_ENCRYPT(i_tarjeta, cliente_id)); #13. INSERTAMOS LA TARJETA DEL CLIENTE
				SELECT LAST_INSERT_ID() INTO tarjeta_id; #13, NOS DEVUELVE EL ULTIMO ID QUE FUE REGISTRADO
                
                INSERT INTO servicios VALUES (0, cliente_id, tarjeta_id, i_suscripcion); # 15.INSERTAMOS A SERVICIOS
                
                SELECT 'Servicio asignado con éxito' INTO o_respuesta; #16. IMPRIMIMOS  Y ALMACENAMOIS EN VARIABLE DE SALIDA O RESPUESTA
			
            END IF;
		
        COMMIT; #4. DEFINIMOS EL FINAL
    
    END //
DELIMITER ;

SELECT * FROM suscripciones;
SELECT * FROM clientes;
SELECT * FROM tarjetas;
SELECT * FROM servicios;
SELECT * FROM actividad_clientes;

# para mostrar los procedures creados
SHOW PROCEDURE STATUS WHERE db = 'curso_sql';
DROP PROCEDURE sp_asignar_servicio; 


# para llamar al procedure asignado, primero se asigna el servicio, nombre, correo, tarjeta, y paramentro de respuesta @res
CALL sp_asignar_servicio(2, 'Kenai', 'kenai@gmail.com', '1234567890123490', @res);
CALL sp_asignar_servicio(1, 'Kenai1', 'eenai1111@gmail.com', '1234567890123491', @res);
CALL sp_asignar_servicio(1, 'rps', 'rsd@gmail.com', '1223167890123491', @res);
CALL sp_asignar_servicio(1, 'rpSSSSSSSSSSs', 'rsSSAAAAASd@gmail.com', '5223167890123411', @res);
SELECT @res;

# para listar sp
SHOW TRIGGERS FROM curso_sql;
DROP TRIGGER tg_actividad_clientes;
DROP TRIGGER sp_asignar_servicio;
/*
SINTAXIS TRIGGERS

DELIMITER //
CREATE TRIGGER nombre_disparador
  [BEFORE | AFTER] [INSERT | UPDATE | DELETE] # SE ESPECIFICA
  ON nombre_tabla
  FOR EACH ROW
BEGIN
END //
DELIMITER ;

*/

# TRIGGERS  O DISPARADORES, SON OBJETOS QUE SE UTILZAN PARA EJECUTAR DE MANERA AUTOMATICA UNA ACCION
# EN RESPUESTA A CIERTOS EVENTOS DENTRO DE LA BASE DE DATOS Y ESTOS EVENTOS SE VAN A DISPARAR
# CUANDO SE LLEGUE A EJECUTAR DENTRO DE LA BASE DE DATOS ALGUNA OPERACION QUE AFECTE DATOS
# ES DECIR UN INSSERT, UPDATE OR DELET, LOS TRIGGERS MANEJAN UNA SINTAXIS  .


CREATE TABLE actividad_clientes(
  ac_id INT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
  cliente INT UNSIGNED,
  fecha DATETIME,
  FOREIGN KEY (cliente)
    REFERENCES clientes(cliente_id)
    ON DELETE RESTRICT
    ON UPDATE CASCADE
);

DELIMITER //

CREATE TRIGGER tg_actividad_clientes
  AFTER INSERT # DESPUES DE QUE SE HAYA CREADO UN SERVICIO
  ON clientes
  FOR EACH ROW # PARA CADA CLIENTE AFECTADO

  BEGIN
    
    INSERT INTO actividad_clientes VALUES (0, NEW.cliente_id, NOW()); # INSERTAR ACTIVIDAD CLIENTES, CON NEW SE OPTINE EL VALOR QUE SE INCERTO EL VALOR DISPARADOR, Y LUEGO LA FECHA CON NOW()
    
  END //

DELIMITER ;

SHOW TRIGGERS FROM curso_sql;
DROP TRIGGER tg_actividad_clientes;





