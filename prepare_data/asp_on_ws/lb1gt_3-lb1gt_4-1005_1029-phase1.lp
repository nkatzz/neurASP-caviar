
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

	orientation(lb1gt_3,134,1005). coords(lb1gt_3,199,80,1005).  visible(lb1gt_3,1005). happensAt(walking(lb1gt_3),1005). orientation(lb1gt_4,90,1005). coords(lb1gt_4,310,112,1005).  visible(lb1gt_4,1005). happensAt(inactive(lb1gt_4),1005).
orientation(lb1gt_3,134,1006). coords(lb1gt_3,198,80,1006).  visible(lb1gt_3,1006). happensAt(walking(lb1gt_3),1006). orientation(lb1gt_4,90,1006). coords(lb1gt_4,310,112,1006).  visible(lb1gt_4,1006). happensAt(inactive(lb1gt_4),1006).
orientation(lb1gt_3,134,1007). coords(lb1gt_3,195,80,1007).  visible(lb1gt_3,1007). happensAt(walking(lb1gt_3),1007). orientation(lb1gt_4,90,1007). coords(lb1gt_4,310,112,1007).  visible(lb1gt_4,1007). happensAt(inactive(lb1gt_4),1007).
orientation(lb1gt_3,134,1008). coords(lb1gt_3,193,80,1008).  visible(lb1gt_3,1008). happensAt(walking(lb1gt_3),1008). orientation(lb1gt_4,90,1008). coords(lb1gt_4,310,112,1008).  visible(lb1gt_4,1008). happensAt(inactive(lb1gt_4),1008).
orientation(lb1gt_3,134,1009). coords(lb1gt_3,189,80,1009).  visible(lb1gt_3,1009). happensAt(walking(lb1gt_3),1009). orientation(lb1gt_4,90,1009). coords(lb1gt_4,310,112,1009).  visible(lb1gt_4,1009). happensAt(inactive(lb1gt_4),1009).
orientation(lb1gt_3,134,1010). coords(lb1gt_3,186,79,1010).  visible(lb1gt_3,1010). happensAt(walking(lb1gt_3),1010). orientation(lb1gt_4,90,1010). coords(lb1gt_4,310,112,1010).  visible(lb1gt_4,1010). happensAt(inactive(lb1gt_4),1010).
orientation(lb1gt_3,134,1011). coords(lb1gt_3,184,79,1011).  visible(lb1gt_3,1011). happensAt(walking(lb1gt_3),1011). orientation(lb1gt_4,90,1011). coords(lb1gt_4,310,112,1011).  visible(lb1gt_4,1011). happensAt(inactive(lb1gt_4),1011).
orientation(lb1gt_3,134,1012). coords(lb1gt_3,182,79,1012).  visible(lb1gt_3,1012). happensAt(walking(lb1gt_3),1012). orientation(lb1gt_4,90,1012). coords(lb1gt_4,310,112,1012).  visible(lb1gt_4,1012). happensAt(inactive(lb1gt_4),1012).
orientation(lb1gt_3,134,1013). coords(lb1gt_3,180,78,1013).  visible(lb1gt_3,1013). happensAt(walking(lb1gt_3),1013). orientation(lb1gt_4,90,1013). coords(lb1gt_4,310,112,1013).  visible(lb1gt_4,1013). happensAt(inactive(lb1gt_4),1013).
orientation(lb1gt_3,134,1014). coords(lb1gt_3,179,78,1014).  visible(lb1gt_3,1014). happensAt(walking(lb1gt_3),1014). orientation(lb1gt_4,90,1014). coords(lb1gt_4,310,112,1014).  visible(lb1gt_4,1014). happensAt(inactive(lb1gt_4),1014).
orientation(lb1gt_3,134,1015). coords(lb1gt_3,179,78,1015).  visible(lb1gt_3,1015). happensAt(walking(lb1gt_3),1015). orientation(lb1gt_4,90,1015). coords(lb1gt_4,310,112,1015).  visible(lb1gt_4,1015). happensAt(inactive(lb1gt_4),1015).
orientation(lb1gt_3,134,1016). coords(lb1gt_3,179,78,1016).  visible(lb1gt_3,1016). happensAt(walking(lb1gt_3),1016). orientation(lb1gt_4,90,1016). coords(lb1gt_4,310,112,1016).  visible(lb1gt_4,1016). happensAt(inactive(lb1gt_4),1016).
orientation(lb1gt_3,134,1017). coords(lb1gt_3,178,78,1017).  visible(lb1gt_3,1017). happensAt(walking(lb1gt_3),1017). orientation(lb1gt_4,90,1017). coords(lb1gt_4,310,112,1017).  visible(lb1gt_4,1017). happensAt(inactive(lb1gt_4),1017).
orientation(lb1gt_3,134,1018). coords(lb1gt_3,177,78,1018).  visible(lb1gt_3,1018). happensAt(walking(lb1gt_3),1018). orientation(lb1gt_4,90,1018). coords(lb1gt_4,310,112,1018).  visible(lb1gt_4,1018). happensAt(inactive(lb1gt_4),1018).
orientation(lb1gt_3,134,1019). coords(lb1gt_3,176,77,1019).  visible(lb1gt_3,1019). happensAt(walking(lb1gt_3),1019). orientation(lb1gt_4,90,1019). coords(lb1gt_4,310,112,1019).  visible(lb1gt_4,1019). happensAt(inactive(lb1gt_4),1019).
orientation(lb1gt_3,134,1020). coords(lb1gt_3,174,77,1020).  visible(lb1gt_3,1020). happensAt(walking(lb1gt_3),1020). orientation(lb1gt_4,90,1020). coords(lb1gt_4,310,112,1020).  visible(lb1gt_4,1020). happensAt(inactive(lb1gt_4),1020).
orientation(lb1gt_3,134,1021). coords(lb1gt_3,172,76,1021).  visible(lb1gt_3,1021). happensAt(walking(lb1gt_3),1021). orientation(lb1gt_4,90,1021). coords(lb1gt_4,310,112,1021).  visible(lb1gt_4,1021). happensAt(inactive(lb1gt_4),1021).
orientation(lb1gt_3,134,1022). coords(lb1gt_3,170,75,1022).  visible(lb1gt_3,1022). happensAt(walking(lb1gt_3),1022). orientation(lb1gt_4,90,1022). coords(lb1gt_4,310,112,1022).  visible(lb1gt_4,1022). happensAt(inactive(lb1gt_4),1022).
orientation(lb1gt_3,134,1023). coords(lb1gt_3,167,74,1023).  visible(lb1gt_3,1023). happensAt(walking(lb1gt_3),1023). orientation(lb1gt_4,90,1023). coords(lb1gt_4,310,112,1023).  visible(lb1gt_4,1023). happensAt(inactive(lb1gt_4),1023).
orientation(lb1gt_3,134,1024). coords(lb1gt_3,166,73,1024).  visible(lb1gt_3,1024). happensAt(walking(lb1gt_3),1024). orientation(lb1gt_4,90,1024). coords(lb1gt_4,310,112,1024).  visible(lb1gt_4,1024). happensAt(inactive(lb1gt_4),1024).
orientation(lb1gt_3,134,1025). coords(lb1gt_3,166,71,1025).  visible(lb1gt_3,1025). happensAt(walking(lb1gt_3),1025). orientation(lb1gt_4,90,1025). coords(lb1gt_4,310,112,1025).  visible(lb1gt_4,1025). happensAt(inactive(lb1gt_4),1025).
orientation(lb1gt_3,134,1026). coords(lb1gt_3,165,70,1026).  visible(lb1gt_3,1026). happensAt(walking(lb1gt_3),1026). orientation(lb1gt_4,90,1026). coords(lb1gt_4,310,112,1026).  visible(lb1gt_4,1026). happensAt(inactive(lb1gt_4),1026).
orientation(lb1gt_3,134,1027). coords(lb1gt_3,164,69,1027).  visible(lb1gt_3,1027). happensAt(walking(lb1gt_3),1027). orientation(lb1gt_4,90,1027). coords(lb1gt_4,310,112,1027).  visible(lb1gt_4,1027). happensAt(inactive(lb1gt_4),1027).
orientation(lb1gt_3,134,1028). coords(lb1gt_3,164,69,1028).  visible(lb1gt_3,1028). happensAt(walking(lb1gt_3),1028). orientation(lb1gt_4,90,1028). coords(lb1gt_4,310,112,1028).  visible(lb1gt_4,1028). happensAt(inactive(lb1gt_4),1028).
orientation(lb1gt_3,134,1029). coords(lb1gt_3,164,69,1029).  visible(lb1gt_3,1029). happensAt(walking(lb1gt_3),1029). orientation(lb1gt_4,90,1029). coords(lb1gt_4,310,112,1029).  visible(lb1gt_4,1029). happensAt(inactive(lb1gt_4),1029).
#show holdsAt/2.