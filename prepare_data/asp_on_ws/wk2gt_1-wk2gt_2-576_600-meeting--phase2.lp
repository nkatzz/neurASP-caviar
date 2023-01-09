

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

	person(wk2gt_1;wk2gt_2).
orientation(wk2gt_1,125,576). coords(wk2gt_1,115,53,576).  visible(wk2gt_1,576). orientation(wk2gt_2,142,576). coords(wk2gt_2,99,63,576).  visible(wk2gt_2,576).
orientation(wk2gt_1,125,577). coords(wk2gt_1,115,53,577).  visible(wk2gt_1,577). orientation(wk2gt_2,142,577). coords(wk2gt_2,99,63,577).  visible(wk2gt_2,577).
orientation(wk2gt_1,125,578). coords(wk2gt_1,115,53,578).  visible(wk2gt_1,578). orientation(wk2gt_2,142,578). coords(wk2gt_2,99,63,578).  visible(wk2gt_2,578).
orientation(wk2gt_1,125,579). coords(wk2gt_1,115,53,579).  visible(wk2gt_1,579). orientation(wk2gt_2,142,579). coords(wk2gt_2,99,63,579).  visible(wk2gt_2,579).
orientation(wk2gt_1,125,580). coords(wk2gt_1,115,53,580).  visible(wk2gt_1,580). orientation(wk2gt_2,142,580). coords(wk2gt_2,99,63,580).  visible(wk2gt_2,580).
orientation(wk2gt_1,125,581). coords(wk2gt_1,115,53,581).  visible(wk2gt_1,581). orientation(wk2gt_2,142,581). coords(wk2gt_2,99,63,581).  visible(wk2gt_2,581).
orientation(wk2gt_1,125,582). coords(wk2gt_1,115,53,582).  visible(wk2gt_1,582). orientation(wk2gt_2,142,582). coords(wk2gt_2,99,63,582).  visible(wk2gt_2,582).
orientation(wk2gt_1,125,583). coords(wk2gt_1,115,53,583).  visible(wk2gt_1,583). orientation(wk2gt_2,142,583). coords(wk2gt_2,99,63,583).  visible(wk2gt_2,583).
orientation(wk2gt_1,125,584). coords(wk2gt_1,115,53,584).  visible(wk2gt_1,584). orientation(wk2gt_2,142,584). coords(wk2gt_2,99,63,584).  visible(wk2gt_2,584).
orientation(wk2gt_1,125,585). coords(wk2gt_1,115,53,585).  visible(wk2gt_1,585). orientation(wk2gt_2,142,585). coords(wk2gt_2,99,63,585).  visible(wk2gt_2,585).
orientation(wk2gt_1,125,586). coords(wk2gt_1,115,53,586).  visible(wk2gt_1,586). orientation(wk2gt_2,142,586). coords(wk2gt_2,99,63,586).  visible(wk2gt_2,586).
orientation(wk2gt_1,125,587). coords(wk2gt_1,115,53,587).  visible(wk2gt_1,587). orientation(wk2gt_2,142,587). coords(wk2gt_2,99,63,587).  visible(wk2gt_2,587).
orientation(wk2gt_1,125,588). coords(wk2gt_1,115,53,588).  visible(wk2gt_1,588). orientation(wk2gt_2,142,588). coords(wk2gt_2,99,63,588).  visible(wk2gt_2,588).
orientation(wk2gt_1,125,589). coords(wk2gt_1,115,53,589).  visible(wk2gt_1,589). orientation(wk2gt_2,142,589). coords(wk2gt_2,99,63,589).  visible(wk2gt_2,589).
orientation(wk2gt_1,125,590). coords(wk2gt_1,115,53,590).  visible(wk2gt_1,590). orientation(wk2gt_2,142,590). coords(wk2gt_2,99,63,590).  visible(wk2gt_2,590).
orientation(wk2gt_1,125,591). coords(wk2gt_1,115,53,591).  visible(wk2gt_1,591). orientation(wk2gt_2,142,591). coords(wk2gt_2,99,63,591).  visible(wk2gt_2,591).
orientation(wk2gt_1,125,592). coords(wk2gt_1,115,53,592).  visible(wk2gt_1,592). orientation(wk2gt_2,142,592). coords(wk2gt_2,99,63,592).  visible(wk2gt_2,592).
orientation(wk2gt_1,125,593). coords(wk2gt_1,115,53,593).  visible(wk2gt_1,593). orientation(wk2gt_2,142,593). coords(wk2gt_2,99,63,593).  visible(wk2gt_2,593).
orientation(wk2gt_1,125,594). coords(wk2gt_1,115,53,594).  visible(wk2gt_1,594). orientation(wk2gt_2,142,594). coords(wk2gt_2,99,63,594).  visible(wk2gt_2,594).
orientation(wk2gt_1,125,595). coords(wk2gt_1,115,53,595).  visible(wk2gt_1,595). orientation(wk2gt_2,142,595). coords(wk2gt_2,99,63,595).  visible(wk2gt_2,595).
orientation(wk2gt_1,125,596). coords(wk2gt_1,115,53,596).  visible(wk2gt_1,596). orientation(wk2gt_2,142,596). coords(wk2gt_2,99,63,596).  visible(wk2gt_2,596).
orientation(wk2gt_1,125,597). coords(wk2gt_1,115,53,597).  visible(wk2gt_1,597). orientation(wk2gt_2,142,597). coords(wk2gt_2,99,63,597).  visible(wk2gt_2,597).
orientation(wk2gt_1,125,598). coords(wk2gt_1,115,53,598).  visible(wk2gt_1,598). orientation(wk2gt_2,142,598). coords(wk2gt_2,99,63,598).  visible(wk2gt_2,598).
orientation(wk2gt_1,125,599). coords(wk2gt_1,115,53,599).  visible(wk2gt_1,599). orientation(wk2gt_2,142,599). coords(wk2gt_2,99,63,599).  visible(wk2gt_2,599).
orientation(wk2gt_1,125,600). coords(wk2gt_1,115,53,600).  visible(wk2gt_1,600). orientation(wk2gt_2,142,600). coords(wk2gt_2,99,63,600).  visible(wk2gt_2,600).
orientation(wk2gt_1,125,601). coords(wk2gt_1,115,53,601).  visible(wk2gt_1,601). orientation(wk2gt_2,142,601). coords(wk2gt_2,99,63,601).  visible(wk2gt_2,601).
happensAt(disappear(wk2gt_1),600). happensAt(disappear(wk2gt_2),600).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=576..600.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=576..600.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),576). :- holdsAt(meeting(wk2gt_2,wk2gt_1),576).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),577). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),577).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),578). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),578).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),579). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),579).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),580). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),580).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),581). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),581).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),582). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),582).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),583). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),583).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),584). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),584).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),585). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),585).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),586). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),586).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),587). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),587).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),588). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),588).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),589). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),589).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),590). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),590).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),591). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),591).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),592). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),592).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),593). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),593).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),594). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),594).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),595). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),595).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),596). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),596).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),597). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),597).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),598). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),598).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),599). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),599).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),600). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),600).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),601). :- holdsAt(meeting(wk2gt_2,wk2gt_1),601).
:- simpleEvent(T,P,walking), T>576 , T<600.
:- simpleEvent(T,P,running), T>576 , T<600.

#show simpleEvent/3.