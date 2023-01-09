

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

	person(lbgt_3;lbgt_4).
orientation(lbgt_3,61,571). coords(lbgt_3,146,162,571).  happensAt(appear(lbgt_3),571). orientation(lbgt_4,174,571). coords(lbgt_4,127,151,571).  visible(lbgt_4,571).
orientation(lbgt_3,61,572). coords(lbgt_3,146,162,572).  visible(lbgt_3,572). orientation(lbgt_4,175,572). coords(lbgt_4,126,151,572).  visible(lbgt_4,572).
orientation(lbgt_3,61,573). coords(lbgt_3,146,162,573).  visible(lbgt_3,573). orientation(lbgt_4,175,573). coords(lbgt_4,125,152,573).  visible(lbgt_4,573).
orientation(lbgt_3,61,574). coords(lbgt_3,146,162,574).  visible(lbgt_3,574). orientation(lbgt_4,171,574). coords(lbgt_4,124,152,574).  visible(lbgt_4,574).
orientation(lbgt_3,61,575). coords(lbgt_3,146,162,575).  visible(lbgt_3,575). orientation(lbgt_4,172,575). coords(lbgt_4,123,152,575).  visible(lbgt_4,575).
orientation(lbgt_3,61,576). coords(lbgt_3,146,162,576).  visible(lbgt_3,576). orientation(lbgt_4,172,576). coords(lbgt_4,123,152,576).  visible(lbgt_4,576).
orientation(lbgt_3,61,577). coords(lbgt_3,146,162,577).  visible(lbgt_3,577). orientation(lbgt_4,170,577). coords(lbgt_4,122,152,577).  visible(lbgt_4,577).
orientation(lbgt_3,61,578). coords(lbgt_3,146,162,578).  visible(lbgt_3,578). orientation(lbgt_4,171,578). coords(lbgt_4,122,152,578).  visible(lbgt_4,578).
orientation(lbgt_3,61,579). coords(lbgt_3,146,162,579).  visible(lbgt_3,579). orientation(lbgt_4,169,579). coords(lbgt_4,122,152,579).  visible(lbgt_4,579).
orientation(lbgt_3,61,580). coords(lbgt_3,146,162,580).  visible(lbgt_3,580). orientation(lbgt_4,169,580). coords(lbgt_4,121,152,580).  visible(lbgt_4,580).
orientation(lbgt_3,61,581). coords(lbgt_3,146,162,581).  visible(lbgt_3,581). orientation(lbgt_4,167,581). coords(lbgt_4,120,152,581).  visible(lbgt_4,581).
orientation(lbgt_3,61,582). coords(lbgt_3,146,162,582).  visible(lbgt_3,582). orientation(lbgt_4,166,582). coords(lbgt_4,120,152,582).  visible(lbgt_4,582).
orientation(lbgt_3,61,583). coords(lbgt_3,146,162,583).  visible(lbgt_3,583). orientation(lbgt_4,166,583). coords(lbgt_4,120,152,583).  visible(lbgt_4,583).
orientation(lbgt_3,61,584). coords(lbgt_3,146,162,584).  visible(lbgt_3,584). orientation(lbgt_4,164,584). coords(lbgt_4,119,152,584).  visible(lbgt_4,584).
orientation(lbgt_3,61,585). coords(lbgt_3,146,162,585).  visible(lbgt_3,585). orientation(lbgt_4,161,585). coords(lbgt_4,119,152,585).  visible(lbgt_4,585).
orientation(lbgt_3,61,586). coords(lbgt_3,146,162,586).  visible(lbgt_3,586). orientation(lbgt_4,161,586). coords(lbgt_4,118,152,586).  visible(lbgt_4,586).
orientation(lbgt_3,61,587). coords(lbgt_3,146,162,587).  visible(lbgt_3,587). orientation(lbgt_4,161,587). coords(lbgt_4,118,152,587).  visible(lbgt_4,587).
orientation(lbgt_3,61,588). coords(lbgt_3,146,162,588).  visible(lbgt_3,588). orientation(lbgt_4,161,588). coords(lbgt_4,118,152,588).  visible(lbgt_4,588).
orientation(lbgt_3,61,589). coords(lbgt_3,146,162,589).  visible(lbgt_3,589). orientation(lbgt_4,161,589). coords(lbgt_4,118,152,589).  visible(lbgt_4,589).
orientation(lbgt_3,61,590). coords(lbgt_3,146,162,590).  visible(lbgt_3,590). orientation(lbgt_4,161,590). coords(lbgt_4,117,152,590).  visible(lbgt_4,590).
orientation(lbgt_3,61,591). coords(lbgt_3,146,162,591).  visible(lbgt_3,591). orientation(lbgt_4,161,591). coords(lbgt_4,117,152,591).  visible(lbgt_4,591).
orientation(lbgt_3,61,592). coords(lbgt_3,146,162,592).  visible(lbgt_3,592). orientation(lbgt_4,161,592). coords(lbgt_4,117,153,592).  visible(lbgt_4,592).
orientation(lbgt_3,61,593). coords(lbgt_3,146,162,593).  visible(lbgt_3,593). orientation(lbgt_4,161,593). coords(lbgt_4,117,153,593).  visible(lbgt_4,593).
orientation(lbgt_3,61,594). coords(lbgt_3,146,162,594).  visible(lbgt_3,594). orientation(lbgt_4,161,594). coords(lbgt_4,117,153,594).  visible(lbgt_4,594).
orientation(lbgt_3,61,595). coords(lbgt_3,146,162,595).  visible(lbgt_3,595). orientation(lbgt_4,161,595). coords(lbgt_4,117,153,595).  visible(lbgt_4,595).
orientation(lbgt_3,61,596). coords(lbgt_3,146,162,596).  visible(lbgt_3,596). orientation(lbgt_4,161,596). coords(lbgt_4,117,153,596).  visible(lbgt_4,596).
happensAt(disappear(lbgt_3),595). happensAt(disappear(lbgt_4),595).
1{simpleEvent(T,lbgt_3,active); simpleEvent(T,lbgt_3,inactive); simpleEvent(T,lbgt_3,walking); simpleEvent(T,lbgt_3,running)}1 :- T=571..595.
1{simpleEvent(T,lbgt_4,active); simpleEvent(T,lbgt_4,inactive); simpleEvent(T,lbgt_4,walking); simpleEvent(T,lbgt_4,running)}1 :- T=571..595.
:- holdsAt(meeting(lbgt_3,lbgt_4),571). :- holdsAt(meeting(lbgt_4,lbgt_3),571).
:- not holdsAt(meeting(lbgt_3,lbgt_4),572). :- not holdsAt(meeting(lbgt_4,lbgt_3),572).
:- not holdsAt(meeting(lbgt_3,lbgt_4),573). :- not holdsAt(meeting(lbgt_4,lbgt_3),573).
:- not holdsAt(meeting(lbgt_3,lbgt_4),574). :- not holdsAt(meeting(lbgt_4,lbgt_3),574).
:- not holdsAt(meeting(lbgt_3,lbgt_4),575). :- not holdsAt(meeting(lbgt_4,lbgt_3),575).
:- not holdsAt(meeting(lbgt_3,lbgt_4),576). :- not holdsAt(meeting(lbgt_4,lbgt_3),576).
:- not holdsAt(meeting(lbgt_3,lbgt_4),577). :- not holdsAt(meeting(lbgt_4,lbgt_3),577).
:- not holdsAt(meeting(lbgt_3,lbgt_4),578). :- not holdsAt(meeting(lbgt_4,lbgt_3),578).
:- not holdsAt(meeting(lbgt_3,lbgt_4),579). :- not holdsAt(meeting(lbgt_4,lbgt_3),579).
:- not holdsAt(meeting(lbgt_3,lbgt_4),580). :- not holdsAt(meeting(lbgt_4,lbgt_3),580).
:- not holdsAt(meeting(lbgt_3,lbgt_4),581). :- not holdsAt(meeting(lbgt_4,lbgt_3),581).
:- not holdsAt(meeting(lbgt_3,lbgt_4),582). :- not holdsAt(meeting(lbgt_4,lbgt_3),582).
:- not holdsAt(meeting(lbgt_3,lbgt_4),583). :- not holdsAt(meeting(lbgt_4,lbgt_3),583).
:- not holdsAt(meeting(lbgt_3,lbgt_4),584). :- not holdsAt(meeting(lbgt_4,lbgt_3),584).
:- not holdsAt(meeting(lbgt_3,lbgt_4),585). :- not holdsAt(meeting(lbgt_4,lbgt_3),585).
:- not holdsAt(meeting(lbgt_3,lbgt_4),586). :- not holdsAt(meeting(lbgt_4,lbgt_3),586).
:- not holdsAt(meeting(lbgt_3,lbgt_4),587). :- not holdsAt(meeting(lbgt_4,lbgt_3),587).
:- not holdsAt(meeting(lbgt_3,lbgt_4),588). :- not holdsAt(meeting(lbgt_4,lbgt_3),588).
:- not holdsAt(meeting(lbgt_3,lbgt_4),589). :- not holdsAt(meeting(lbgt_4,lbgt_3),589).
:- not holdsAt(meeting(lbgt_3,lbgt_4),590). :- not holdsAt(meeting(lbgt_4,lbgt_3),590).
:- not holdsAt(meeting(lbgt_3,lbgt_4),591). :- not holdsAt(meeting(lbgt_4,lbgt_3),591).
:- not holdsAt(meeting(lbgt_3,lbgt_4),592). :- not holdsAt(meeting(lbgt_4,lbgt_3),592).
:- not holdsAt(meeting(lbgt_3,lbgt_4),593). :- not holdsAt(meeting(lbgt_4,lbgt_3),593).
:- not holdsAt(meeting(lbgt_3,lbgt_4),594). :- not holdsAt(meeting(lbgt_4,lbgt_3),594).
:- not holdsAt(meeting(lbgt_3,lbgt_4),595). :- not holdsAt(meeting(lbgt_4,lbgt_3),595).
:- holdsAt(meeting(lbgt_3,lbgt_4),596). :- holdsAt(meeting(lbgt_4,lbgt_3),596).
:- simpleEvent(T,P,walking), T>571 , T<595.
:- simpleEvent(T,P,running), T>571 , T<595.

#show simpleEvent/3.