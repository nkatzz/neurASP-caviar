

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
orientation(mwt2gt_1,153,591). coords(mwt2gt_1,29,155,591).  visible(mwt2gt_1,591). orientation(mwt2gt_2,152,591). coords(mwt2gt_2,26,165,591).  visible(mwt2gt_2,591).
orientation(mwt2gt_1,153,592). coords(mwt2gt_1,29,155,592).  visible(mwt2gt_1,592). orientation(mwt2gt_2,152,592). coords(mwt2gt_2,26,165,592).  visible(mwt2gt_2,592).
orientation(mwt2gt_1,153,593). coords(mwt2gt_1,29,155,593).  visible(mwt2gt_1,593). orientation(mwt2gt_2,152,593). coords(mwt2gt_2,26,165,593).  visible(mwt2gt_2,593).
orientation(mwt2gt_1,153,594). coords(mwt2gt_1,29,155,594).  visible(mwt2gt_1,594). orientation(mwt2gt_2,152,594). coords(mwt2gt_2,26,165,594).  visible(mwt2gt_2,594).
orientation(mwt2gt_1,153,595). coords(mwt2gt_1,29,155,595).  visible(mwt2gt_1,595). orientation(mwt2gt_2,152,595). coords(mwt2gt_2,26,165,595).  visible(mwt2gt_2,595).
orientation(mwt2gt_1,153,596). coords(mwt2gt_1,29,155,596).  visible(mwt2gt_1,596). orientation(mwt2gt_2,152,596). coords(mwt2gt_2,26,165,596).  visible(mwt2gt_2,596).
orientation(mwt2gt_1,153,597). coords(mwt2gt_1,29,155,597).  visible(mwt2gt_1,597). orientation(mwt2gt_2,152,597). coords(mwt2gt_2,26,165,597).  visible(mwt2gt_2,597).
orientation(mwt2gt_1,153,598). coords(mwt2gt_1,28,155,598).  visible(mwt2gt_1,598). orientation(mwt2gt_2,152,598). coords(mwt2gt_2,27,164,598).  visible(mwt2gt_2,598).
orientation(mwt2gt_1,153,599). coords(mwt2gt_1,29,155,599).  visible(mwt2gt_1,599). orientation(mwt2gt_2,152,599). coords(mwt2gt_2,27,164,599).  visible(mwt2gt_2,599).
orientation(mwt2gt_1,153,600). coords(mwt2gt_1,28,156,600).  visible(mwt2gt_1,600). orientation(mwt2gt_2,152,600). coords(mwt2gt_2,27,164,600).  visible(mwt2gt_2,600).
orientation(mwt2gt_1,153,601). coords(mwt2gt_1,28,156,601).  visible(mwt2gt_1,601). orientation(mwt2gt_2,152,601). coords(mwt2gt_2,26,164,601).  visible(mwt2gt_2,601).
orientation(mwt2gt_1,153,602). coords(mwt2gt_1,28,156,602).  visible(mwt2gt_1,602). orientation(mwt2gt_2,152,602). coords(mwt2gt_2,26,164,602).  visible(mwt2gt_2,602).
happensAt(disappear(mwt2gt_1),601). happensAt(disappear(mwt2gt_2),601).
1{simpleEvent(T,mwt2gt_1,active); simpleEvent(T,mwt2gt_1,inactive); simpleEvent(T,mwt2gt_1,walking); simpleEvent(T,mwt2gt_1,running)}1 :- T=577..601.
1{simpleEvent(T,mwt2gt_2,active); simpleEvent(T,mwt2gt_2,inactive); simpleEvent(T,mwt2gt_2,walking); simpleEvent(T,mwt2gt_2,running)}1 :- T=577..601.
:- holdsAt(moving(mwt2gt_1,mwt2gt_2),577). :- holdsAt(moving(mwt2gt_2,mwt2gt_1),577).
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
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),590). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),590).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),591). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),591).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),592). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),592).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),593). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),593).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),594). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),594).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),595). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),595).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),596). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),596).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),597). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),597).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),598). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),598).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),599). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),599).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),600). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),600).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),601). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),601).
:- holdsAt(moving(mwt2gt_1,mwt2gt_2),602). :- holdsAt(moving(mwt2gt_2,mwt2gt_1),602).
:- simpleEvent(T,P,inactive), T>577 , T<601.
:- simpleEvent(T,P,running), T>577 , T<601.

#show simpleEvent/3.