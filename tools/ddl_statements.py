all_stars_ddl = '''
        CREATE TABLE IF NOT EXISTS all_stars
        (
          id                                        INT AUTO_INCREMENT PRIMARY KEY ,
          player_id                                   VARCHAR(12) ,
          year_id                                     VARCHAR(5) ,
          game_num                                    INT ,
          game_id                                     VARCHAR(15) ,
          team_id                                     VARCHAR(5) ,
          league_id                                   VARCHAR(2) ,
          go                                          INT , 
          starting_pos                                INT
          );'''



