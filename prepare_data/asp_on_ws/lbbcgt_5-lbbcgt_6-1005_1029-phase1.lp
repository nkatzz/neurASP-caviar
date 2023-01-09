
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(lbbcgt_5,154,1005). coords(lbbcgt_5,179,117,1005).  visible(lbbcgt_5,1005). happensAt(active(lbbcgt_5),1005). orientation(lbbcgt_6,151,1005). coords(lbbcgt_6,64,94,1005).  visible(lbbcgt_6,1005). happensAt(walking(lbbcgt_6),1005).
orientation(lbbcgt_5,154,1006). coords(lbbcgt_5,178,117,1006).  visible(lbbcgt_5,1006). happensAt(active(lbbcgt_5),1006). orientation(lbbcgt_6,151,1006). coords(lbbcgt_6,65,94,1006).  visible(lbbcgt_6,1006). happensAt(walking(lbbcgt_6),1006).
orientation(lbbcgt_5,154,1007). coords(lbbcgt_5,178,117,1007).  visible(lbbcgt_5,1007). happensAt(active(lbbcgt_5),1007). orientation(lbbcgt_6,151,1007). coords(lbbcgt_6,65,94,1007).  visible(lbbcgt_6,1007). happensAt(walking(lbbcgt_6),1007).
orientation(lbbcgt_5,154,1008). coords(lbbcgt_5,178,117,1008).  visible(lbbcgt_5,1008). happensAt(active(lbbcgt_5),1008). orientation(lbbcgt_6,151,1008). coords(lbbcgt_6,64,94,1008).  visible(lbbcgt_6,1008). happensAt(walking(lbbcgt_6),1008).
orientation(lbbcgt_5,154,1009). coords(lbbcgt_5,178,117,1009).  visible(lbbcgt_5,1009). happensAt(active(lbbcgt_5),1009). orientation(lbbcgt_6,151,1009). coords(lbbcgt_6,63,94,1009).  visible(lbbcgt_6,1009). happensAt(walking(lbbcgt_6),1009).
orientation(lbbcgt_5,154,1010). coords(lbbcgt_5,178,117,1010).  visible(lbbcgt_5,1010). happensAt(active(lbbcgt_5),1010). orientation(lbbcgt_6,151,1010). coords(lbbcgt_6,63,94,1010).  visible(lbbcgt_6,1010). happensAt(walking(lbbcgt_6),1010).
orientation(lbbcgt_5,154,1011). coords(lbbcgt_5,178,117,1011).  visible(lbbcgt_5,1011). happensAt(active(lbbcgt_5),1011). orientation(lbbcgt_6,151,1011). coords(lbbcgt_6,63,94,1011).  visible(lbbcgt_6,1011). happensAt(walking(lbbcgt_6),1011).
orientation(lbbcgt_5,154,1012). coords(lbbcgt_5,178,117,1012).  visible(lbbcgt_5,1012). happensAt(active(lbbcgt_5),1012). orientation(lbbcgt_6,151,1012). coords(lbbcgt_6,62,94,1012).  visible(lbbcgt_6,1012). happensAt(walking(lbbcgt_6),1012).
orientation(lbbcgt_5,154,1013). coords(lbbcgt_5,179,117,1013).  visible(lbbcgt_5,1013). happensAt(active(lbbcgt_5),1013). orientation(lbbcgt_6,151,1013). coords(lbbcgt_6,63,94,1013).  visible(lbbcgt_6,1013). happensAt(walking(lbbcgt_6),1013).
orientation(lbbcgt_5,154,1014). coords(lbbcgt_5,178,117,1014).  visible(lbbcgt_5,1014). happensAt(active(lbbcgt_5),1014). orientation(lbbcgt_6,151,1014). coords(lbbcgt_6,63,94,1014).  visible(lbbcgt_6,1014). happensAt(walking(lbbcgt_6),1014).
orientation(lbbcgt_5,154,1015). coords(lbbcgt_5,178,117,1015).  visible(lbbcgt_5,1015). happensAt(active(lbbcgt_5),1015). orientation(lbbcgt_6,151,1015). coords(lbbcgt_6,63,94,1015).  visible(lbbcgt_6,1015). happensAt(walking(lbbcgt_6),1015).
orientation(lbbcgt_5,154,1016). coords(lbbcgt_5,178,117,1016).  visible(lbbcgt_5,1016). happensAt(active(lbbcgt_5),1016). orientation(lbbcgt_6,151,1016). coords(lbbcgt_6,63,94,1016).  visible(lbbcgt_6,1016). happensAt(walking(lbbcgt_6),1016).
orientation(lbbcgt_5,154,1017). coords(lbbcgt_5,178,117,1017).  visible(lbbcgt_5,1017). happensAt(active(lbbcgt_5),1017). orientation(lbbcgt_6,151,1017). coords(lbbcgt_6,62,94,1017).  visible(lbbcgt_6,1017). happensAt(walking(lbbcgt_6),1017).
orientation(lbbcgt_5,154,1018). coords(lbbcgt_5,179,117,1018).  visible(lbbcgt_5,1018). happensAt(active(lbbcgt_5),1018). orientation(lbbcgt_6,151,1018). coords(lbbcgt_6,62,94,1018).  visible(lbbcgt_6,1018). happensAt(walking(lbbcgt_6),1018).
orientation(lbbcgt_5,154,1019). coords(lbbcgt_5,179,117,1019).  visible(lbbcgt_5,1019). happensAt(active(lbbcgt_5),1019). orientation(lbbcgt_6,151,1019). coords(lbbcgt_6,62,94,1019).  visible(lbbcgt_6,1019). happensAt(walking(lbbcgt_6),1019).
orientation(lbbcgt_5,154,1020). coords(lbbcgt_5,180,117,1020).  visible(lbbcgt_5,1020). happensAt(active(lbbcgt_5),1020). orientation(lbbcgt_6,151,1020). coords(lbbcgt_6,62,94,1020).  visible(lbbcgt_6,1020). happensAt(walking(lbbcgt_6),1020).
orientation(lbbcgt_5,154,1021). coords(lbbcgt_5,181,117,1021).  visible(lbbcgt_5,1021). happensAt(active(lbbcgt_5),1021). orientation(lbbcgt_6,151,1021). coords(lbbcgt_6,62,94,1021).  visible(lbbcgt_6,1021). happensAt(walking(lbbcgt_6),1021).
orientation(lbbcgt_5,154,1022). coords(lbbcgt_5,183,117,1022).  visible(lbbcgt_5,1022). happensAt(active(lbbcgt_5),1022). orientation(lbbcgt_6,151,1022). coords(lbbcgt_6,62,94,1022).  visible(lbbcgt_6,1022). happensAt(walking(lbbcgt_6),1022).
orientation(lbbcgt_5,154,1023). coords(lbbcgt_5,184,117,1023).  visible(lbbcgt_5,1023). happensAt(active(lbbcgt_5),1023). orientation(lbbcgt_6,151,1023). coords(lbbcgt_6,62,94,1023).  visible(lbbcgt_6,1023). happensAt(walking(lbbcgt_6),1023).
orientation(lbbcgt_5,154,1024). coords(lbbcgt_5,185,117,1024).  visible(lbbcgt_5,1024). happensAt(active(lbbcgt_5),1024). orientation(lbbcgt_6,151,1024). coords(lbbcgt_6,62,94,1024).  visible(lbbcgt_6,1024). happensAt(walking(lbbcgt_6),1024).
orientation(lbbcgt_5,154,1025). coords(lbbcgt_5,185,117,1025).  visible(lbbcgt_5,1025). happensAt(active(lbbcgt_5),1025). orientation(lbbcgt_6,151,1025). coords(lbbcgt_6,62,94,1025).  visible(lbbcgt_6,1025). happensAt(walking(lbbcgt_6),1025).
orientation(lbbcgt_5,154,1026). coords(lbbcgt_5,185,117,1026).  visible(lbbcgt_5,1026). happensAt(active(lbbcgt_5),1026). orientation(lbbcgt_6,151,1026). coords(lbbcgt_6,62,94,1026).  visible(lbbcgt_6,1026). happensAt(walking(lbbcgt_6),1026).
orientation(lbbcgt_5,154,1027). coords(lbbcgt_5,185,117,1027).  visible(lbbcgt_5,1027). happensAt(active(lbbcgt_5),1027). orientation(lbbcgt_6,151,1027). coords(lbbcgt_6,62,94,1027).  visible(lbbcgt_6,1027). happensAt(walking(lbbcgt_6),1027).
orientation(lbbcgt_5,154,1028). coords(lbbcgt_5,185,117,1028).  visible(lbbcgt_5,1028). happensAt(active(lbbcgt_5),1028). orientation(lbbcgt_6,151,1028). coords(lbbcgt_6,62,95,1028).  visible(lbbcgt_6,1028). happensAt(walking(lbbcgt_6),1028).
orientation(lbbcgt_5,154,1029). coords(lbbcgt_5,185,117,1029).  visible(lbbcgt_5,1029). happensAt(active(lbbcgt_5),1029). orientation(lbbcgt_6,151,1029). coords(lbbcgt_6,62,95,1029).  visible(lbbcgt_6,1029). happensAt(walking(lbbcgt_6),1029).
#show holdsAt/2.