
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

	orientation(lb2gt_5,152,1006). coords(lb2gt_5,135,143,1006).  visible(lb2gt_5,1006). happensAt(walking(lb2gt_5),1006). orientation(lb2gt_7,0,1006). coords(lb2gt_7,98,263,1006).  visible(lb2gt_7,1006). happensAt(walking(lb2gt_7),1006).
orientation(lb2gt_5,152,1007). coords(lb2gt_5,137,144,1007).  visible(lb2gt_5,1007). happensAt(walking(lb2gt_5),1007). orientation(lb2gt_7,0,1007). coords(lb2gt_7,97,264,1007).  visible(lb2gt_7,1007). happensAt(walking(lb2gt_7),1007).
orientation(lb2gt_5,152,1008). coords(lb2gt_5,137,144,1008).  visible(lb2gt_5,1008). happensAt(walking(lb2gt_5),1008). orientation(lb2gt_7,0,1008). coords(lb2gt_7,97,264,1008).  visible(lb2gt_7,1008). happensAt(walking(lb2gt_7),1008).
orientation(lb2gt_5,152,1009). coords(lb2gt_5,138,145,1009).  visible(lb2gt_5,1009). happensAt(walking(lb2gt_5),1009). orientation(lb2gt_7,0,1009). coords(lb2gt_7,97,264,1009).  visible(lb2gt_7,1009). happensAt(walking(lb2gt_7),1009).
orientation(lb2gt_5,148,1010). coords(lb2gt_5,138,145,1010).  visible(lb2gt_5,1010). happensAt(walking(lb2gt_5),1010). orientation(lb2gt_7,0,1010). coords(lb2gt_7,96,264,1010).  visible(lb2gt_7,1010). happensAt(walking(lb2gt_7),1010).
orientation(lb2gt_5,148,1011). coords(lb2gt_5,139,145,1011).  visible(lb2gt_5,1011). happensAt(walking(lb2gt_5),1011). orientation(lb2gt_7,0,1011). coords(lb2gt_7,96,265,1011).  visible(lb2gt_7,1011). happensAt(walking(lb2gt_7),1011).
orientation(lb2gt_5,147,1012). coords(lb2gt_5,139,145,1012).  visible(lb2gt_5,1012). happensAt(walking(lb2gt_5),1012). orientation(lb2gt_7,0,1012). coords(lb2gt_7,96,265,1012).  visible(lb2gt_7,1012). happensAt(walking(lb2gt_7),1012).
orientation(lb2gt_5,146,1013). coords(lb2gt_5,139,145,1013).  visible(lb2gt_5,1013). happensAt(walking(lb2gt_5),1013). orientation(lb2gt_7,0,1013). coords(lb2gt_7,96,265,1013).  visible(lb2gt_7,1013). happensAt(walking(lb2gt_7),1013).
orientation(lb2gt_5,145,1014). coords(lb2gt_5,142,145,1014).  visible(lb2gt_5,1014). happensAt(walking(lb2gt_5),1014). orientation(lb2gt_7,0,1014). coords(lb2gt_7,95,266,1014).  visible(lb2gt_7,1014). happensAt(walking(lb2gt_7),1014).
orientation(lb2gt_5,148,1015). coords(lb2gt_5,144,146,1015).  visible(lb2gt_5,1015). happensAt(walking(lb2gt_5),1015). orientation(lb2gt_7,0,1015). coords(lb2gt_7,95,266,1015).  visible(lb2gt_7,1015). happensAt(walking(lb2gt_7),1015).
orientation(lb2gt_5,148,1016). coords(lb2gt_5,147,146,1016).  visible(lb2gt_5,1016). happensAt(walking(lb2gt_5),1016). orientation(lb2gt_7,0,1016). coords(lb2gt_7,95,267,1016).  visible(lb2gt_7,1016). happensAt(walking(lb2gt_7),1016).
orientation(lb2gt_5,146,1017). coords(lb2gt_5,149,147,1017).  visible(lb2gt_5,1017). happensAt(walking(lb2gt_5),1017). orientation(lb2gt_7,0,1017). coords(lb2gt_7,94,267,1017).  visible(lb2gt_7,1017). happensAt(walking(lb2gt_7),1017).
orientation(lb2gt_5,146,1018). coords(lb2gt_5,151,148,1018).  visible(lb2gt_5,1018). happensAt(walking(lb2gt_5),1018). orientation(lb2gt_7,0,1018). coords(lb2gt_7,94,268,1018).  visible(lb2gt_7,1018). happensAt(walking(lb2gt_7),1018).
orientation(lb2gt_5,145,1019). coords(lb2gt_5,152,148,1019).  visible(lb2gt_5,1019). happensAt(walking(lb2gt_5),1019). orientation(lb2gt_7,0,1019). coords(lb2gt_7,94,268,1019).  visible(lb2gt_7,1019). happensAt(walking(lb2gt_7),1019).
orientation(lb2gt_5,146,1020). coords(lb2gt_5,154,150,1020).  visible(lb2gt_5,1020). happensAt(walking(lb2gt_5),1020). orientation(lb2gt_7,0,1020). coords(lb2gt_7,94,268,1020).  visible(lb2gt_7,1020). happensAt(walking(lb2gt_7),1020).
orientation(lb2gt_5,147,1021). coords(lb2gt_5,155,150,1021).  visible(lb2gt_5,1021). happensAt(walking(lb2gt_5),1021). orientation(lb2gt_7,0,1021). coords(lb2gt_7,94,269,1021).  visible(lb2gt_7,1021). happensAt(walking(lb2gt_7),1021).
orientation(lb2gt_5,151,1022). coords(lb2gt_5,156,151,1022).  visible(lb2gt_5,1022). happensAt(walking(lb2gt_5),1022). orientation(lb2gt_7,0,1022). coords(lb2gt_7,94,269,1022).  visible(lb2gt_7,1022). happensAt(walking(lb2gt_7),1022).
orientation(lb2gt_5,152,1023). coords(lb2gt_5,156,151,1023).  visible(lb2gt_5,1023). happensAt(walking(lb2gt_5),1023). orientation(lb2gt_7,0,1023). coords(lb2gt_7,94,269,1023).  visible(lb2gt_7,1023). happensAt(walking(lb2gt_7),1023).
orientation(lb2gt_5,152,1024). coords(lb2gt_5,157,152,1024).  visible(lb2gt_5,1024). happensAt(walking(lb2gt_5),1024). orientation(lb2gt_7,0,1024). coords(lb2gt_7,94,269,1024).  visible(lb2gt_7,1024). happensAt(walking(lb2gt_7),1024).
orientation(lb2gt_5,152,1025). coords(lb2gt_5,157,152,1025).  visible(lb2gt_5,1025). happensAt(walking(lb2gt_5),1025). orientation(lb2gt_7,0,1025). coords(lb2gt_7,93,268,1025).  visible(lb2gt_7,1025). happensAt(walking(lb2gt_7),1025).
orientation(lb2gt_5,152,1026). coords(lb2gt_5,157,152,1026).  visible(lb2gt_5,1026). happensAt(walking(lb2gt_5),1026). orientation(lb2gt_7,0,1026). coords(lb2gt_7,93,268,1026).  visible(lb2gt_7,1026). happensAt(walking(lb2gt_7),1026).
orientation(lb2gt_5,152,1027). coords(lb2gt_5,157,153,1027).  visible(lb2gt_5,1027). happensAt(walking(lb2gt_5),1027). orientation(lb2gt_7,0,1027). coords(lb2gt_7,93,269,1027).  visible(lb2gt_7,1027). happensAt(walking(lb2gt_7),1027).
orientation(lb2gt_5,152,1028). coords(lb2gt_5,159,154,1028).  visible(lb2gt_5,1028). happensAt(walking(lb2gt_5),1028). orientation(lb2gt_7,0,1028). coords(lb2gt_7,93,269,1028).  visible(lb2gt_7,1028). happensAt(walking(lb2gt_7),1028).
orientation(lb2gt_5,152,1029). coords(lb2gt_5,161,156,1029).  visible(lb2gt_5,1029). happensAt(walking(lb2gt_5),1029). orientation(lb2gt_7,0,1029). coords(lb2gt_7,93,268,1029).  visible(lb2gt_7,1029). happensAt(walking(lb2gt_7),1029).
orientation(lb2gt_5,152,1030). coords(lb2gt_5,164,158,1030).  visible(lb2gt_5,1030). happensAt(walking(lb2gt_5),1030). orientation(lb2gt_7,0,1030). coords(lb2gt_7,89,268,1030).  visible(lb2gt_7,1030). happensAt(walking(lb2gt_7),1030).
#show holdsAt/2.