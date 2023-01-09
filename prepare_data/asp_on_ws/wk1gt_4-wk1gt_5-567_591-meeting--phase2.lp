

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

	person(wk1gt_4;wk1gt_5).
orientation(wk1gt_4,157,567). coords(wk1gt_4,56,170,567).  visible(wk1gt_4,567). orientation(wk1gt_5,162,567). coords(wk1gt_5,57,150,567).  visible(wk1gt_5,567).
orientation(wk1gt_4,157,568). coords(wk1gt_4,56,170,568).  visible(wk1gt_4,568). orientation(wk1gt_5,162,568). coords(wk1gt_5,57,150,568).  visible(wk1gt_5,568).
orientation(wk1gt_4,157,569). coords(wk1gt_4,56,170,569).  visible(wk1gt_4,569). orientation(wk1gt_5,162,569). coords(wk1gt_5,57,150,569).  visible(wk1gt_5,569).
orientation(wk1gt_4,157,570). coords(wk1gt_4,56,170,570).  visible(wk1gt_4,570). orientation(wk1gt_5,162,570). coords(wk1gt_5,57,150,570).  visible(wk1gt_5,570).
orientation(wk1gt_4,157,571). coords(wk1gt_4,56,170,571).  visible(wk1gt_4,571). orientation(wk1gt_5,162,571). coords(wk1gt_5,57,150,571).  visible(wk1gt_5,571).
orientation(wk1gt_4,157,572). coords(wk1gt_4,56,170,572).  visible(wk1gt_4,572). orientation(wk1gt_5,162,572). coords(wk1gt_5,57,150,572).  visible(wk1gt_5,572).
orientation(wk1gt_4,157,573). coords(wk1gt_4,56,170,573).  visible(wk1gt_4,573). orientation(wk1gt_5,162,573). coords(wk1gt_5,57,150,573).  visible(wk1gt_5,573).
orientation(wk1gt_4,157,574). coords(wk1gt_4,56,170,574).  visible(wk1gt_4,574). orientation(wk1gt_5,162,574). coords(wk1gt_5,57,150,574).  visible(wk1gt_5,574).
orientation(wk1gt_4,157,575). coords(wk1gt_4,56,170,575).  visible(wk1gt_4,575). orientation(wk1gt_5,162,575). coords(wk1gt_5,57,150,575).  visible(wk1gt_5,575).
orientation(wk1gt_4,157,576). coords(wk1gt_4,56,170,576).  visible(wk1gt_4,576). orientation(wk1gt_5,162,576). coords(wk1gt_5,57,150,576).  visible(wk1gt_5,576).
orientation(wk1gt_4,157,577). coords(wk1gt_4,56,170,577).  visible(wk1gt_4,577). orientation(wk1gt_5,162,577). coords(wk1gt_5,57,150,577).  visible(wk1gt_5,577).
orientation(wk1gt_4,157,578). coords(wk1gt_4,56,170,578).  visible(wk1gt_4,578). orientation(wk1gt_5,162,578). coords(wk1gt_5,57,150,578).  visible(wk1gt_5,578).
orientation(wk1gt_4,157,579). coords(wk1gt_4,56,170,579).  visible(wk1gt_4,579). orientation(wk1gt_5,162,579). coords(wk1gt_5,57,150,579).  visible(wk1gt_5,579).
orientation(wk1gt_4,157,580). coords(wk1gt_4,56,170,580).  visible(wk1gt_4,580). orientation(wk1gt_5,162,580). coords(wk1gt_5,57,150,580).  visible(wk1gt_5,580).
orientation(wk1gt_4,157,581). coords(wk1gt_4,56,170,581).  visible(wk1gt_4,581). orientation(wk1gt_5,162,581). coords(wk1gt_5,57,150,581).  visible(wk1gt_5,581).
orientation(wk1gt_4,157,582). coords(wk1gt_4,56,170,582).  visible(wk1gt_4,582). orientation(wk1gt_5,162,582). coords(wk1gt_5,57,150,582).  visible(wk1gt_5,582).
orientation(wk1gt_4,157,583). coords(wk1gt_4,56,170,583).  visible(wk1gt_4,583). orientation(wk1gt_5,162,583). coords(wk1gt_5,57,150,583).  visible(wk1gt_5,583).
orientation(wk1gt_4,157,584). coords(wk1gt_4,56,170,584).  visible(wk1gt_4,584). orientation(wk1gt_5,162,584). coords(wk1gt_5,57,150,584).  visible(wk1gt_5,584).
orientation(wk1gt_4,157,585). coords(wk1gt_4,56,170,585).  visible(wk1gt_4,585). orientation(wk1gt_5,162,585). coords(wk1gt_5,57,150,585).  visible(wk1gt_5,585).
orientation(wk1gt_4,157,586). coords(wk1gt_4,56,170,586).  visible(wk1gt_4,586). orientation(wk1gt_5,162,586). coords(wk1gt_5,57,150,586).  visible(wk1gt_5,586).
orientation(wk1gt_4,157,587). coords(wk1gt_4,56,170,587).  visible(wk1gt_4,587). orientation(wk1gt_5,162,587). coords(wk1gt_5,57,150,587).  visible(wk1gt_5,587).
orientation(wk1gt_4,157,588). coords(wk1gt_4,56,170,588).  visible(wk1gt_4,588). orientation(wk1gt_5,162,588). coords(wk1gt_5,57,150,588).  visible(wk1gt_5,588).
orientation(wk1gt_4,157,589). coords(wk1gt_4,56,170,589).  visible(wk1gt_4,589). orientation(wk1gt_5,162,589). coords(wk1gt_5,57,150,589).  visible(wk1gt_5,589).
orientation(wk1gt_4,157,590). coords(wk1gt_4,56,170,590).  visible(wk1gt_4,590). orientation(wk1gt_5,162,590). coords(wk1gt_5,57,150,590).  visible(wk1gt_5,590).
orientation(wk1gt_4,157,591). coords(wk1gt_4,56,170,591).  visible(wk1gt_4,591). orientation(wk1gt_5,162,591). coords(wk1gt_5,57,150,591).  visible(wk1gt_5,591).
orientation(wk1gt_4,157,592). coords(wk1gt_4,56,170,592).  visible(wk1gt_4,592). orientation(wk1gt_5,162,592). coords(wk1gt_5,57,150,592).  visible(wk1gt_5,592).
happensAt(disappear(wk1gt_4),591). happensAt(disappear(wk1gt_5),591).
1{simpleEvent(T,wk1gt_4,active); simpleEvent(T,wk1gt_4,inactive); simpleEvent(T,wk1gt_4,walking); simpleEvent(T,wk1gt_4,running)}1 :- T=567..591.
1{simpleEvent(T,wk1gt_5,active); simpleEvent(T,wk1gt_5,inactive); simpleEvent(T,wk1gt_5,walking); simpleEvent(T,wk1gt_5,running)}1 :- T=567..591.
:- holdsAt(meeting(wk1gt_4,wk1gt_5),567). :- holdsAt(meeting(wk1gt_5,wk1gt_4),567).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),568). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),568).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),569). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),569).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),570). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),570).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),571). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),571).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),572). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),572).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),573). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),573).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),574). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),574).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),575). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),575).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),576). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),576).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),577). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),577).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),578). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),578).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),579). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),579).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),580). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),580).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),581). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),581).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),582). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),582).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),583). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),583).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),584). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),584).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),585). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),585).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),586). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),586).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),587). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),587).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),588). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),588).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),589). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),589).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),590). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),590).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),591). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),591).
:- holdsAt(meeting(wk1gt_4,wk1gt_5),592). :- holdsAt(meeting(wk1gt_5,wk1gt_4),592).
:- simpleEvent(T,P,walking), T>567 , T<591.
:- simpleEvent(T,P,running), T>567 , T<591.

#show simpleEvent/3.