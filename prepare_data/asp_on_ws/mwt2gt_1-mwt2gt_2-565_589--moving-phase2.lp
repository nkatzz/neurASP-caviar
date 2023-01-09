

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
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).


happensAt(active(X),T) :- simpleEvent(T,X,active).
happensAt(inactive(X),T) :- simpleEvent(T,X,inactive).
happensAt(walking(X),T) :- simpleEvent(T,X,walking).
happensAt(running(X),T) :- simpleEvent(T,X,running). 


holdsByInertia(moving(X,Y),T+1) :- holdsAt(moving(X,Y),T+1), not initiatedAt(moving(X,Y),T).
holdsByInertia(meeting(X,Y),T+1) :- holdsAt(meeting(X,Y),T+1), not initiatedAt(meeting(X,Y),T).

:- holdsByInertia(moving(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2. 
:- holdsByInertia(meeting(X,Y),T+1), simpleEvent(T,X,E1), simpleEvent(T+1,X,E2), E1 != E2.

	person(mwt2gt_1;mwt2gt_2).
orientation(mwt2gt_1,153,565). coords(mwt2gt_1,36,155,565).  visible(mwt2gt_1,565). orientation(mwt2gt_2,152,565). coords(mwt2gt_2,31,167,565).  visible(mwt2gt_2,565).
orientation(mwt2gt_1,153,566). coords(mwt2gt_1,35,155,566).  visible(mwt2gt_1,566). orientation(mwt2gt_2,152,566). coords(mwt2gt_2,31,167,566).  visible(mwt2gt_2,566).
orientation(mwt2gt_1,153,567). coords(mwt2gt_1,35,155,567).  visible(mwt2gt_1,567). orientation(mwt2gt_2,152,567). coords(mwt2gt_2,31,167,567).  visible(mwt2gt_2,567).
orientation(mwt2gt_1,153,568). coords(mwt2gt_1,34,155,568).  visible(mwt2gt_1,568). orientation(mwt2gt_2,152,568). coords(mwt2gt_2,29,167,568).  visible(mwt2gt_2,568).
orientation(mwt2gt_1,153,569). coords(mwt2gt_1,34,155,569).  visible(mwt2gt_1,569). orientation(mwt2gt_2,152,569). coords(mwt2gt_2,29,167,569).  visible(mwt2gt_2,569).
orientation(mwt2gt_1,153,570). coords(mwt2gt_1,33,155,570).  visible(mwt2gt_1,570). orientation(mwt2gt_2,152,570). coords(mwt2gt_2,29,167,570).  visible(mwt2gt_2,570).
orientation(mwt2gt_1,153,571). coords(mwt2gt_1,33,155,571).  visible(mwt2gt_1,571). orientation(mwt2gt_2,152,571). coords(mwt2gt_2,29,167,571).  visible(mwt2gt_2,571).
orientation(mwt2gt_1,153,572). coords(mwt2gt_1,33,155,572).  visible(mwt2gt_1,572). orientation(mwt2gt_2,152,572). coords(mwt2gt_2,29,167,572).  visible(mwt2gt_2,572).
orientation(mwt2gt_1,153,573). coords(mwt2gt_1,33,155,573).  visible(mwt2gt_1,573). orientation(mwt2gt_2,152,573). coords(mwt2gt_2,29,167,573).  visible(mwt2gt_2,573).
orientation(mwt2gt_1,153,574). coords(mwt2gt_1,33,155,574).  visible(mwt2gt_1,574). orientation(mwt2gt_2,152,574). coords(mwt2gt_2,28,167,574).  visible(mwt2gt_2,574).
orientation(mwt2gt_1,153,575). coords(mwt2gt_1,33,155,575).  visible(mwt2gt_1,575). orientation(mwt2gt_2,152,575). coords(mwt2gt_2,28,166,575).  visible(mwt2gt_2,575).
orientation(mwt2gt_1,153,576). coords(mwt2gt_1,33,155,576).  visible(mwt2gt_1,576). orientation(mwt2gt_2,152,576). coords(mwt2gt_2,28,166,576).  visible(mwt2gt_2,576).
orientation(mwt2gt_1,153,577). coords(mwt2gt_1,32,155,577).  visible(mwt2gt_1,577). orientation(mwt2gt_2,152,577). coords(mwt2gt_2,28,166,577).  visible(mwt2gt_2,577).
orientation(mwt2gt_1,153,578). coords(mwt2gt_1,32,155,578).  visible(mwt2gt_1,578). orientation(mwt2gt_2,152,578). coords(mwt2gt_2,28,166,578).  visible(mwt2gt_2,578).
orientation(mwt2gt_1,153,579). coords(mwt2gt_1,32,155,579).  visible(mwt2gt_1,579). orientation(mwt2gt_2,152,579). coords(mwt2gt_2,27,166,579).  visible(mwt2gt_2,579).
orientation(mwt2gt_1,153,580). coords(mwt2gt_1,32,155,580).  visible(mwt2gt_1,580). orientation(mwt2gt_2,152,580). coords(mwt2gt_2,27,166,580).  visible(mwt2gt_2,580).
orientation(mwt2gt_1,153,581). coords(mwt2gt_1,31,155,581).  visible(mwt2gt_1,581). orientation(mwt2gt_2,152,581). coords(mwt2gt_2,27,166,581).  visible(mwt2gt_2,581).
orientation(mwt2gt_1,153,582). coords(mwt2gt_1,31,155,582).  visible(mwt2gt_1,582). orientation(mwt2gt_2,152,582). coords(mwt2gt_2,26,166,582).  visible(mwt2gt_2,582).
orientation(mwt2gt_1,153,583). coords(mwt2gt_1,31,155,583).  visible(mwt2gt_1,583). orientation(mwt2gt_2,152,583). coords(mwt2gt_2,26,166,583).  visible(mwt2gt_2,583).
orientation(mwt2gt_1,153,584). coords(mwt2gt_1,31,155,584).  visible(mwt2gt_1,584). orientation(mwt2gt_2,152,584). coords(mwt2gt_2,26,166,584).  visible(mwt2gt_2,584).
orientation(mwt2gt_1,153,585). coords(mwt2gt_1,31,155,585).  visible(mwt2gt_1,585). orientation(mwt2gt_2,152,585). coords(mwt2gt_2,26,166,585).  visible(mwt2gt_2,585).
orientation(mwt2gt_1,153,586). coords(mwt2gt_1,31,155,586).  visible(mwt2gt_1,586). orientation(mwt2gt_2,152,586). coords(mwt2gt_2,26,166,586).  visible(mwt2gt_2,586).
orientation(mwt2gt_1,153,587). coords(mwt2gt_1,31,155,587).  visible(mwt2gt_1,587). orientation(mwt2gt_2,152,587). coords(mwt2gt_2,26,166,587).  visible(mwt2gt_2,587).
orientation(mwt2gt_1,153,588). coords(mwt2gt_1,31,155,588).  visible(mwt2gt_1,588). orientation(mwt2gt_2,152,588). coords(mwt2gt_2,26,166,588).  visible(mwt2gt_2,588).
orientation(mwt2gt_1,153,589). coords(mwt2gt_1,31,155,589).  visible(mwt2gt_1,589). orientation(mwt2gt_2,152,589). coords(mwt2gt_2,26,165,589).  visible(mwt2gt_2,589).
orientation(mwt2gt_1,153,590). coords(mwt2gt_1,31,155,590).  visible(mwt2gt_1,590). orientation(mwt2gt_2,152,590). coords(mwt2gt_2,26,165,590).  visible(mwt2gt_2,590).
happensAt(disappear(mwt2gt_1),589). happensAt(disappear(mwt2gt_2),589).
1{simpleEvent(T,mwt2gt_1,active); simpleEvent(T,mwt2gt_1,inactive); simpleEvent(T,mwt2gt_1,walking); simpleEvent(T,mwt2gt_1,running)}1 :- T=565..589.
1{simpleEvent(T,mwt2gt_2,active); simpleEvent(T,mwt2gt_2,inactive); simpleEvent(T,mwt2gt_2,walking); simpleEvent(T,mwt2gt_2,running)}1 :- T=565..589.
:- holdsAt(moving(mwt2gt_1,mwt2gt_2),565). :- holdsAt(moving(mwt2gt_2,mwt2gt_1),565).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),566). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),566).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),567). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),567).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),568). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),568).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),569). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),569).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),570). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),570).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),571). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),571).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),572). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),572).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),573). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),573).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),574). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),574).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),575). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),575).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),576). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),576).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),577). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),577).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),578). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),578).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),579). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),579).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),580). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),580).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),581). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),581).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),582). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),582).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),583). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),583).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),584). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),584).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),585). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),585).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),586). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),586).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),587). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),587).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),588). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),588).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),589). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),589).
:- holdsAt(moving(mwt2gt_1,mwt2gt_2),590). :- holdsAt(moving(mwt2gt_2,mwt2gt_1),590).
:- simpleEvent(T,P,inactive), T>565 , T<589.
:- simpleEvent(T,P,running), T>565 , T<589.

#show simpleEvent/3.