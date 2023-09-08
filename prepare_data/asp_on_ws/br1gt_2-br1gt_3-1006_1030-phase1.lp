
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

	orientation(br1gt_2,174,1006). coords(br1gt_2,110,214,1006).  visible(br1gt_2,1006). happensAt(walking(br1gt_2),1006). orientation(br1gt_3,128,1006). coords(br1gt_3,256,97,1006).  visible(br1gt_3,1006). happensAt(walking(br1gt_3),1006).
orientation(br1gt_2,174,1007). coords(br1gt_2,111,214,1007).  visible(br1gt_2,1007). happensAt(walking(br1gt_2),1007). orientation(br1gt_3,128,1007). coords(br1gt_3,256,97,1007).  visible(br1gt_3,1007). happensAt(walking(br1gt_3),1007).
orientation(br1gt_2,174,1008). coords(br1gt_2,110,213,1008).  visible(br1gt_2,1008). happensAt(walking(br1gt_2),1008). orientation(br1gt_3,128,1008). coords(br1gt_3,254,97,1008).  visible(br1gt_3,1008). happensAt(walking(br1gt_3),1008).
orientation(br1gt_2,174,1009). coords(br1gt_2,110,213,1009).  visible(br1gt_2,1009). happensAt(walking(br1gt_2),1009). orientation(br1gt_3,128,1009). coords(br1gt_3,253,97,1009).  visible(br1gt_3,1009). happensAt(walking(br1gt_3),1009).
orientation(br1gt_2,174,1010). coords(br1gt_2,111,215,1010).  visible(br1gt_2,1010). happensAt(walking(br1gt_2),1010). orientation(br1gt_3,135,1010). coords(br1gt_3,252,97,1010).  visible(br1gt_3,1010). happensAt(walking(br1gt_3),1010).
orientation(br1gt_2,174,1011). coords(br1gt_2,111,215,1011).  visible(br1gt_2,1011). happensAt(walking(br1gt_2),1011). orientation(br1gt_3,135,1011). coords(br1gt_3,250,96,1011).  visible(br1gt_3,1011). happensAt(walking(br1gt_3),1011).
orientation(br1gt_2,174,1012). coords(br1gt_2,111,215,1012).  visible(br1gt_2,1012). happensAt(walking(br1gt_2),1012). orientation(br1gt_3,135,1012). coords(br1gt_3,249,96,1012).  visible(br1gt_3,1012). happensAt(walking(br1gt_3),1012).
orientation(br1gt_2,174,1013). coords(br1gt_2,111,216,1013).  visible(br1gt_2,1013). happensAt(walking(br1gt_2),1013). orientation(br1gt_3,135,1013). coords(br1gt_3,245,96,1013).  visible(br1gt_3,1013). happensAt(walking(br1gt_3),1013).
orientation(br1gt_2,174,1014). coords(br1gt_2,111,216,1014).  visible(br1gt_2,1014). happensAt(walking(br1gt_2),1014). orientation(br1gt_3,135,1014). coords(br1gt_3,243,94,1014).  visible(br1gt_3,1014). happensAt(walking(br1gt_3),1014).
orientation(br1gt_2,174,1015). coords(br1gt_2,111,215,1015).  visible(br1gt_2,1015). happensAt(walking(br1gt_2),1015). orientation(br1gt_3,135,1015). coords(br1gt_3,240,94,1015).  visible(br1gt_3,1015). happensAt(walking(br1gt_3),1015).
orientation(br1gt_2,174,1016). coords(br1gt_2,111,216,1016).  visible(br1gt_2,1016). happensAt(walking(br1gt_2),1016). orientation(br1gt_3,135,1016). coords(br1gt_3,237,95,1016).  visible(br1gt_3,1016). happensAt(walking(br1gt_3),1016).
orientation(br1gt_2,174,1017). coords(br1gt_2,111,214,1017).  visible(br1gt_2,1017). happensAt(walking(br1gt_2),1017). orientation(br1gt_3,135,1017). coords(br1gt_3,235,95,1017).  visible(br1gt_3,1017). happensAt(walking(br1gt_3),1017).
orientation(br1gt_2,174,1018). coords(br1gt_2,111,214,1018).  visible(br1gt_2,1018). happensAt(walking(br1gt_2),1018). orientation(br1gt_3,135,1018). coords(br1gt_3,233,95,1018).  visible(br1gt_3,1018). happensAt(walking(br1gt_3),1018).
orientation(br1gt_2,174,1019). coords(br1gt_2,111,215,1019).  visible(br1gt_2,1019). happensAt(walking(br1gt_2),1019). orientation(br1gt_3,135,1019). coords(br1gt_3,233,95,1019).  visible(br1gt_3,1019). happensAt(walking(br1gt_3),1019).
orientation(br1gt_2,174,1020). coords(br1gt_2,111,215,1020).  visible(br1gt_2,1020). happensAt(active(br1gt_2),1020). orientation(br1gt_3,135,1020). coords(br1gt_3,231,95,1020).  visible(br1gt_3,1020). happensAt(walking(br1gt_3),1020).
orientation(br1gt_2,174,1021). coords(br1gt_2,111,215,1021).  visible(br1gt_2,1021). happensAt(active(br1gt_2),1021). orientation(br1gt_3,135,1021). coords(br1gt_3,231,96,1021).  visible(br1gt_3,1021). happensAt(walking(br1gt_3),1021).
orientation(br1gt_2,174,1022). coords(br1gt_2,111,216,1022).  visible(br1gt_2,1022). happensAt(active(br1gt_2),1022). orientation(br1gt_3,135,1022). coords(br1gt_3,230,96,1022).  visible(br1gt_3,1022). happensAt(walking(br1gt_3),1022).
orientation(br1gt_2,174,1023). coords(br1gt_2,111,216,1023).  visible(br1gt_2,1023). happensAt(active(br1gt_2),1023). orientation(br1gt_3,135,1023). coords(br1gt_3,229,96,1023).  visible(br1gt_3,1023). happensAt(walking(br1gt_3),1023).
orientation(br1gt_2,174,1024). coords(br1gt_2,111,216,1024).  visible(br1gt_2,1024). happensAt(active(br1gt_2),1024). orientation(br1gt_3,135,1024). coords(br1gt_3,228,96,1024).  visible(br1gt_3,1024). happensAt(walking(br1gt_3),1024).
orientation(br1gt_2,174,1025). coords(br1gt_2,111,216,1025).  visible(br1gt_2,1025). happensAt(active(br1gt_2),1025). orientation(br1gt_3,137,1025). coords(br1gt_3,226,96,1025).  visible(br1gt_3,1025). happensAt(walking(br1gt_3),1025).
orientation(br1gt_2,174,1026). coords(br1gt_2,111,216,1026).  visible(br1gt_2,1026). happensAt(active(br1gt_2),1026). orientation(br1gt_3,137,1026). coords(br1gt_3,224,96,1026).  visible(br1gt_3,1026). happensAt(walking(br1gt_3),1026).
orientation(br1gt_2,174,1027). coords(br1gt_2,111,216,1027).  visible(br1gt_2,1027). happensAt(active(br1gt_2),1027). orientation(br1gt_3,137,1027). coords(br1gt_3,221,96,1027).  visible(br1gt_3,1027). happensAt(walking(br1gt_3),1027).
orientation(br1gt_2,174,1028). coords(br1gt_2,111,216,1028).  visible(br1gt_2,1028). happensAt(active(br1gt_2),1028). orientation(br1gt_3,137,1028). coords(br1gt_3,219,96,1028).  visible(br1gt_3,1028). happensAt(walking(br1gt_3),1028).
orientation(br1gt_2,174,1029). coords(br1gt_2,111,216,1029).  visible(br1gt_2,1029). happensAt(active(br1gt_2),1029). orientation(br1gt_3,137,1029). coords(br1gt_3,217,97,1029).  visible(br1gt_3,1029). happensAt(walking(br1gt_3),1029).
orientation(br1gt_2,174,1030). coords(br1gt_2,111,219,1030).  visible(br1gt_2,1030). happensAt(active(br1gt_2),1030). orientation(br1gt_3,137,1030). coords(br1gt_3,214,97,1030).  visible(br1gt_3,1030). happensAt(walking(br1gt_3),1030).
#show holdsAt/2.