
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

	orientation(wk2gt_1,133,1008). coords(wk2gt_1,135,39,1008).  visible(wk2gt_1,1008). happensAt(walking(wk2gt_1),1008). orientation(wk2gt_2,140,1008). coords(wk2gt_2,97,65,1008).  visible(wk2gt_2,1008). happensAt(inactive(wk2gt_2),1008).
orientation(wk2gt_1,133,1009). coords(wk2gt_1,135,39,1009).  visible(wk2gt_1,1009). happensAt(walking(wk2gt_1),1009). orientation(wk2gt_2,140,1009). coords(wk2gt_2,97,65,1009).  visible(wk2gt_2,1009). happensAt(inactive(wk2gt_2),1009).
orientation(wk2gt_1,133,1010). coords(wk2gt_1,135,39,1010).  visible(wk2gt_1,1010). happensAt(walking(wk2gt_1),1010). orientation(wk2gt_2,140,1010). coords(wk2gt_2,97,65,1010).  visible(wk2gt_2,1010). happensAt(inactive(wk2gt_2),1010).
orientation(wk2gt_1,133,1011). coords(wk2gt_1,135,39,1011).  visible(wk2gt_1,1011). happensAt(walking(wk2gt_1),1011). orientation(wk2gt_2,140,1011). coords(wk2gt_2,97,65,1011).  visible(wk2gt_2,1011). happensAt(inactive(wk2gt_2),1011).
orientation(wk2gt_1,133,1012). coords(wk2gt_1,135,39,1012).  visible(wk2gt_1,1012). happensAt(walking(wk2gt_1),1012). orientation(wk2gt_2,140,1012). coords(wk2gt_2,97,65,1012).  visible(wk2gt_2,1012). happensAt(inactive(wk2gt_2),1012).
orientation(wk2gt_1,133,1013). coords(wk2gt_1,135,39,1013).  visible(wk2gt_1,1013). happensAt(walking(wk2gt_1),1013). orientation(wk2gt_2,140,1013). coords(wk2gt_2,97,65,1013).  visible(wk2gt_2,1013). happensAt(inactive(wk2gt_2),1013).
orientation(wk2gt_1,133,1014). coords(wk2gt_1,135,39,1014).  visible(wk2gt_1,1014). happensAt(walking(wk2gt_1),1014). orientation(wk2gt_2,140,1014). coords(wk2gt_2,97,65,1014).  visible(wk2gt_2,1014). happensAt(inactive(wk2gt_2),1014).
orientation(wk2gt_1,133,1015). coords(wk2gt_1,135,39,1015).  visible(wk2gt_1,1015). happensAt(walking(wk2gt_1),1015). orientation(wk2gt_2,140,1015). coords(wk2gt_2,97,65,1015).  visible(wk2gt_2,1015). happensAt(inactive(wk2gt_2),1015).
orientation(wk2gt_1,133,1016). coords(wk2gt_1,136,39,1016).  visible(wk2gt_1,1016). happensAt(walking(wk2gt_1),1016). orientation(wk2gt_2,140,1016). coords(wk2gt_2,97,65,1016).  visible(wk2gt_2,1016). happensAt(inactive(wk2gt_2),1016).
orientation(wk2gt_1,133,1017). coords(wk2gt_1,136,39,1017).  visible(wk2gt_1,1017). happensAt(walking(wk2gt_1),1017). orientation(wk2gt_2,140,1017). coords(wk2gt_2,97,65,1017).  visible(wk2gt_2,1017). happensAt(inactive(wk2gt_2),1017).
orientation(wk2gt_1,133,1018). coords(wk2gt_1,136,39,1018).  visible(wk2gt_1,1018). happensAt(walking(wk2gt_1),1018). orientation(wk2gt_2,140,1018). coords(wk2gt_2,97,65,1018).  visible(wk2gt_2,1018). happensAt(inactive(wk2gt_2),1018).
orientation(wk2gt_1,133,1019). coords(wk2gt_1,136,39,1019).  visible(wk2gt_1,1019). happensAt(walking(wk2gt_1),1019). orientation(wk2gt_2,140,1019). coords(wk2gt_2,97,65,1019).  visible(wk2gt_2,1019). happensAt(inactive(wk2gt_2),1019).
orientation(wk2gt_1,133,1020). coords(wk2gt_1,136,39,1020).  visible(wk2gt_1,1020). happensAt(walking(wk2gt_1),1020). orientation(wk2gt_2,140,1020). coords(wk2gt_2,97,65,1020).  visible(wk2gt_2,1020). happensAt(inactive(wk2gt_2),1020).
orientation(wk2gt_1,133,1021). coords(wk2gt_1,137,39,1021).  visible(wk2gt_1,1021). happensAt(walking(wk2gt_1),1021). orientation(wk2gt_2,140,1021). coords(wk2gt_2,97,65,1021).  visible(wk2gt_2,1021). happensAt(inactive(wk2gt_2),1021).
orientation(wk2gt_1,133,1022). coords(wk2gt_1,137,39,1022).  visible(wk2gt_1,1022). happensAt(walking(wk2gt_1),1022). orientation(wk2gt_2,140,1022). coords(wk2gt_2,98,65,1022).  visible(wk2gt_2,1022). happensAt(inactive(wk2gt_2),1022).
orientation(wk2gt_1,133,1023). coords(wk2gt_1,137,39,1023).  visible(wk2gt_1,1023). happensAt(walking(wk2gt_1),1023). orientation(wk2gt_2,140,1023). coords(wk2gt_2,98,65,1023).  visible(wk2gt_2,1023). happensAt(inactive(wk2gt_2),1023).
orientation(wk2gt_1,133,1024). coords(wk2gt_1,137,39,1024).  visible(wk2gt_1,1024). happensAt(walking(wk2gt_1),1024). orientation(wk2gt_2,140,1024). coords(wk2gt_2,98,65,1024).  visible(wk2gt_2,1024). happensAt(inactive(wk2gt_2),1024).
orientation(wk2gt_1,133,1025). coords(wk2gt_1,137,39,1025).  visible(wk2gt_1,1025). happensAt(walking(wk2gt_1),1025). orientation(wk2gt_2,140,1025). coords(wk2gt_2,98,65,1025).  visible(wk2gt_2,1025). happensAt(inactive(wk2gt_2),1025).
orientation(wk2gt_1,133,1026). coords(wk2gt_1,137,39,1026).  visible(wk2gt_1,1026). happensAt(walking(wk2gt_1),1026). orientation(wk2gt_2,140,1026). coords(wk2gt_2,98,65,1026).  visible(wk2gt_2,1026). happensAt(inactive(wk2gt_2),1026).
orientation(wk2gt_1,133,1027). coords(wk2gt_1,138,39,1027).  visible(wk2gt_1,1027). happensAt(walking(wk2gt_1),1027). orientation(wk2gt_2,140,1027). coords(wk2gt_2,98,65,1027).  visible(wk2gt_2,1027). happensAt(inactive(wk2gt_2),1027).
orientation(wk2gt_1,133,1028). coords(wk2gt_1,138,38,1028).  visible(wk2gt_1,1028). happensAt(walking(wk2gt_1),1028). orientation(wk2gt_2,140,1028). coords(wk2gt_2,99,64,1028).  visible(wk2gt_2,1028). happensAt(inactive(wk2gt_2),1028).
orientation(wk2gt_1,133,1029). coords(wk2gt_1,138,38,1029).  visible(wk2gt_1,1029). happensAt(walking(wk2gt_1),1029). orientation(wk2gt_2,140,1029). coords(wk2gt_2,99,64,1029).  visible(wk2gt_2,1029). happensAt(inactive(wk2gt_2),1029).
orientation(wk2gt_1,133,1030). coords(wk2gt_1,138,38,1030).  visible(wk2gt_1,1030). happensAt(walking(wk2gt_1),1030). orientation(wk2gt_2,140,1030). coords(wk2gt_2,99,64,1030).  visible(wk2gt_2,1030). happensAt(inactive(wk2gt_2),1030).
orientation(wk2gt_1,133,1031). coords(wk2gt_1,138,38,1031).  visible(wk2gt_1,1031). happensAt(walking(wk2gt_1),1031). orientation(wk2gt_2,140,1031). coords(wk2gt_2,99,64,1031).  visible(wk2gt_2,1031). happensAt(inactive(wk2gt_2),1031).
orientation(wk2gt_1,133,1032). coords(wk2gt_1,139,37,1032).  visible(wk2gt_1,1032). happensAt(walking(wk2gt_1),1032). orientation(wk2gt_2,140,1032). coords(wk2gt_2,99,64,1032).  visible(wk2gt_2,1032). happensAt(inactive(wk2gt_2),1032).
#show holdsAt/2.