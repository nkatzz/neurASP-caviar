
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

	orientation(wk2gt_2,140,1004). coords(wk2gt_2,97,65,1004).  visible(wk2gt_2,1004). happensAt(inactive(wk2gt_2),1004). orientation(wk2gt_5,150,1004). coords(wk2gt_5,112,116,1004).  visible(wk2gt_5,1004). happensAt(walking(wk2gt_5),1004).
orientation(wk2gt_2,140,1005). coords(wk2gt_2,97,65,1005).  visible(wk2gt_2,1005). happensAt(inactive(wk2gt_2),1005). orientation(wk2gt_5,150,1005). coords(wk2gt_5,113,116,1005).  visible(wk2gt_5,1005). happensAt(walking(wk2gt_5),1005).
orientation(wk2gt_2,140,1006). coords(wk2gt_2,97,65,1006).  visible(wk2gt_2,1006). happensAt(inactive(wk2gt_2),1006). orientation(wk2gt_5,150,1006). coords(wk2gt_5,113,116,1006).  visible(wk2gt_5,1006). happensAt(walking(wk2gt_5),1006).
orientation(wk2gt_2,140,1007). coords(wk2gt_2,97,65,1007).  visible(wk2gt_2,1007). happensAt(inactive(wk2gt_2),1007). orientation(wk2gt_5,150,1007). coords(wk2gt_5,113,116,1007).  visible(wk2gt_5,1007). happensAt(walking(wk2gt_5),1007).
orientation(wk2gt_2,140,1008). coords(wk2gt_2,97,65,1008).  visible(wk2gt_2,1008). happensAt(inactive(wk2gt_2),1008). orientation(wk2gt_5,150,1008). coords(wk2gt_5,113,118,1008).  visible(wk2gt_5,1008). happensAt(walking(wk2gt_5),1008).
orientation(wk2gt_2,140,1009). coords(wk2gt_2,97,65,1009).  visible(wk2gt_2,1009). happensAt(inactive(wk2gt_2),1009). orientation(wk2gt_5,150,1009). coords(wk2gt_5,113,118,1009).  visible(wk2gt_5,1009). happensAt(walking(wk2gt_5),1009).
orientation(wk2gt_2,140,1010). coords(wk2gt_2,97,65,1010).  visible(wk2gt_2,1010). happensAt(inactive(wk2gt_2),1010). orientation(wk2gt_5,150,1010). coords(wk2gt_5,114,120,1010).  visible(wk2gt_5,1010). happensAt(walking(wk2gt_5),1010).
orientation(wk2gt_2,140,1011). coords(wk2gt_2,97,65,1011).  visible(wk2gt_2,1011). happensAt(inactive(wk2gt_2),1011). orientation(wk2gt_5,150,1011). coords(wk2gt_5,114,121,1011).  visible(wk2gt_5,1011). happensAt(walking(wk2gt_5),1011).
orientation(wk2gt_2,140,1012). coords(wk2gt_2,97,65,1012).  visible(wk2gt_2,1012). happensAt(inactive(wk2gt_2),1012). orientation(wk2gt_5,150,1012). coords(wk2gt_5,116,122,1012).  visible(wk2gt_5,1012). happensAt(walking(wk2gt_5),1012).
orientation(wk2gt_2,140,1013). coords(wk2gt_2,97,65,1013).  visible(wk2gt_2,1013). happensAt(inactive(wk2gt_2),1013). orientation(wk2gt_5,150,1013). coords(wk2gt_5,118,122,1013).  visible(wk2gt_5,1013). happensAt(walking(wk2gt_5),1013).
orientation(wk2gt_2,140,1014). coords(wk2gt_2,97,65,1014).  visible(wk2gt_2,1014). happensAt(inactive(wk2gt_2),1014). orientation(wk2gt_5,150,1014). coords(wk2gt_5,119,123,1014).  visible(wk2gt_5,1014). happensAt(walking(wk2gt_5),1014).
orientation(wk2gt_2,140,1015). coords(wk2gt_2,97,65,1015).  visible(wk2gt_2,1015). happensAt(inactive(wk2gt_2),1015). orientation(wk2gt_5,150,1015). coords(wk2gt_5,120,123,1015).  visible(wk2gt_5,1015). happensAt(walking(wk2gt_5),1015).
orientation(wk2gt_2,140,1016). coords(wk2gt_2,97,65,1016).  visible(wk2gt_2,1016). happensAt(inactive(wk2gt_2),1016). orientation(wk2gt_5,150,1016). coords(wk2gt_5,122,124,1016).  visible(wk2gt_5,1016). happensAt(walking(wk2gt_5),1016).
orientation(wk2gt_2,140,1017). coords(wk2gt_2,97,65,1017).  visible(wk2gt_2,1017). happensAt(inactive(wk2gt_2),1017). orientation(wk2gt_5,150,1017). coords(wk2gt_5,122,125,1017).  visible(wk2gt_5,1017). happensAt(walking(wk2gt_5),1017).
orientation(wk2gt_2,140,1018). coords(wk2gt_2,97,65,1018).  visible(wk2gt_2,1018). happensAt(inactive(wk2gt_2),1018). orientation(wk2gt_5,150,1018). coords(wk2gt_5,123,125,1018).  visible(wk2gt_5,1018). happensAt(walking(wk2gt_5),1018).
orientation(wk2gt_2,140,1019). coords(wk2gt_2,97,65,1019).  visible(wk2gt_2,1019). happensAt(inactive(wk2gt_2),1019). orientation(wk2gt_5,150,1019). coords(wk2gt_5,123,125,1019).  visible(wk2gt_5,1019). happensAt(walking(wk2gt_5),1019).
orientation(wk2gt_2,140,1020). coords(wk2gt_2,97,65,1020).  visible(wk2gt_2,1020). happensAt(inactive(wk2gt_2),1020). orientation(wk2gt_5,150,1020). coords(wk2gt_5,123,125,1020).  visible(wk2gt_5,1020). happensAt(walking(wk2gt_5),1020).
orientation(wk2gt_2,140,1021). coords(wk2gt_2,97,65,1021).  visible(wk2gt_2,1021). happensAt(inactive(wk2gt_2),1021). orientation(wk2gt_5,150,1021). coords(wk2gt_5,123,125,1021).  visible(wk2gt_5,1021). happensAt(walking(wk2gt_5),1021).
orientation(wk2gt_2,140,1022). coords(wk2gt_2,98,65,1022).  visible(wk2gt_2,1022). happensAt(inactive(wk2gt_2),1022). orientation(wk2gt_5,150,1022). coords(wk2gt_5,124,125,1022).  visible(wk2gt_5,1022). happensAt(walking(wk2gt_5),1022).
orientation(wk2gt_2,140,1023). coords(wk2gt_2,98,65,1023).  visible(wk2gt_2,1023). happensAt(inactive(wk2gt_2),1023). orientation(wk2gt_5,150,1023). coords(wk2gt_5,127,125,1023).  visible(wk2gt_5,1023). happensAt(walking(wk2gt_5),1023).
orientation(wk2gt_2,140,1024). coords(wk2gt_2,98,65,1024).  visible(wk2gt_2,1024). happensAt(inactive(wk2gt_2),1024). orientation(wk2gt_5,150,1024). coords(wk2gt_5,129,125,1024).  visible(wk2gt_5,1024). happensAt(walking(wk2gt_5),1024).
orientation(wk2gt_2,140,1025). coords(wk2gt_2,98,65,1025).  visible(wk2gt_2,1025). happensAt(inactive(wk2gt_2),1025). orientation(wk2gt_5,150,1025). coords(wk2gt_5,131,125,1025).  visible(wk2gt_5,1025). happensAt(walking(wk2gt_5),1025).
orientation(wk2gt_2,140,1026). coords(wk2gt_2,98,65,1026).  visible(wk2gt_2,1026). happensAt(inactive(wk2gt_2),1026). orientation(wk2gt_5,150,1026). coords(wk2gt_5,132,126,1026).  visible(wk2gt_5,1026). happensAt(walking(wk2gt_5),1026).
orientation(wk2gt_2,140,1027). coords(wk2gt_2,98,65,1027).  visible(wk2gt_2,1027). happensAt(inactive(wk2gt_2),1027). orientation(wk2gt_5,150,1027). coords(wk2gt_5,134,126,1027).  visible(wk2gt_5,1027). happensAt(walking(wk2gt_5),1027).
orientation(wk2gt_2,140,1028). coords(wk2gt_2,99,64,1028).  visible(wk2gt_2,1028). happensAt(inactive(wk2gt_2),1028). orientation(wk2gt_5,150,1028). coords(wk2gt_5,135,127,1028).  visible(wk2gt_5,1028). happensAt(walking(wk2gt_5),1028).
#show holdsAt/2.