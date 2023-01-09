

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
orientation(wk2gt_1,125,804). coords(wk2gt_1,115,53,804).  visible(wk2gt_1,804). orientation(wk2gt_2,142,804). coords(wk2gt_2,99,63,804).  visible(wk2gt_2,804).
orientation(wk2gt_1,125,805). coords(wk2gt_1,115,53,805).  visible(wk2gt_1,805). orientation(wk2gt_2,142,805). coords(wk2gt_2,99,63,805).  visible(wk2gt_2,805).
orientation(wk2gt_1,125,806). coords(wk2gt_1,115,53,806).  visible(wk2gt_1,806). orientation(wk2gt_2,142,806). coords(wk2gt_2,99,63,806).  visible(wk2gt_2,806).
orientation(wk2gt_1,125,807). coords(wk2gt_1,115,53,807).  visible(wk2gt_1,807). orientation(wk2gt_2,142,807). coords(wk2gt_2,99,63,807).  visible(wk2gt_2,807).
orientation(wk2gt_1,125,808). coords(wk2gt_1,115,53,808).  visible(wk2gt_1,808). orientation(wk2gt_2,142,808). coords(wk2gt_2,99,63,808).  visible(wk2gt_2,808).
orientation(wk2gt_1,125,809). coords(wk2gt_1,115,53,809).  visible(wk2gt_1,809). orientation(wk2gt_2,142,809). coords(wk2gt_2,99,63,809).  visible(wk2gt_2,809).
orientation(wk2gt_1,125,810). coords(wk2gt_1,115,53,810).  visible(wk2gt_1,810). orientation(wk2gt_2,142,810). coords(wk2gt_2,99,63,810).  visible(wk2gt_2,810).
orientation(wk2gt_1,125,811). coords(wk2gt_1,115,53,811).  visible(wk2gt_1,811). orientation(wk2gt_2,142,811). coords(wk2gt_2,99,63,811).  visible(wk2gt_2,811).
orientation(wk2gt_1,125,812). coords(wk2gt_1,115,53,812).  visible(wk2gt_1,812). orientation(wk2gt_2,142,812). coords(wk2gt_2,99,63,812).  visible(wk2gt_2,812).
orientation(wk2gt_1,125,813). coords(wk2gt_1,115,53,813).  visible(wk2gt_1,813). orientation(wk2gt_2,142,813). coords(wk2gt_2,99,63,813).  visible(wk2gt_2,813).
orientation(wk2gt_1,125,814). coords(wk2gt_1,115,53,814).  visible(wk2gt_1,814). orientation(wk2gt_2,142,814). coords(wk2gt_2,99,63,814).  visible(wk2gt_2,814).
orientation(wk2gt_1,125,815). coords(wk2gt_1,115,53,815).  visible(wk2gt_1,815). orientation(wk2gt_2,142,815). coords(wk2gt_2,99,63,815).  visible(wk2gt_2,815).
orientation(wk2gt_1,125,816). coords(wk2gt_1,115,53,816).  visible(wk2gt_1,816). orientation(wk2gt_2,142,816). coords(wk2gt_2,99,63,816).  visible(wk2gt_2,816).
orientation(wk2gt_1,125,817). coords(wk2gt_1,115,53,817).  visible(wk2gt_1,817). orientation(wk2gt_2,142,817). coords(wk2gt_2,99,63,817).  visible(wk2gt_2,817).
orientation(wk2gt_1,125,818). coords(wk2gt_1,115,53,818).  visible(wk2gt_1,818). orientation(wk2gt_2,142,818). coords(wk2gt_2,99,63,818).  visible(wk2gt_2,818).
orientation(wk2gt_1,125,819). coords(wk2gt_1,115,53,819).  visible(wk2gt_1,819). orientation(wk2gt_2,142,819). coords(wk2gt_2,99,63,819).  visible(wk2gt_2,819).
orientation(wk2gt_1,125,820). coords(wk2gt_1,115,53,820).  visible(wk2gt_1,820). orientation(wk2gt_2,142,820). coords(wk2gt_2,99,63,820).  visible(wk2gt_2,820).
orientation(wk2gt_1,125,821). coords(wk2gt_1,115,53,821).  visible(wk2gt_1,821). orientation(wk2gt_2,142,821). coords(wk2gt_2,99,63,821).  visible(wk2gt_2,821).
orientation(wk2gt_1,125,822). coords(wk2gt_1,115,53,822).  visible(wk2gt_1,822). orientation(wk2gt_2,142,822). coords(wk2gt_2,99,63,822).  visible(wk2gt_2,822).
orientation(wk2gt_1,125,823). coords(wk2gt_1,115,53,823).  visible(wk2gt_1,823). orientation(wk2gt_2,142,823). coords(wk2gt_2,99,63,823).  visible(wk2gt_2,823).
orientation(wk2gt_1,125,824). coords(wk2gt_1,115,53,824).  visible(wk2gt_1,824). orientation(wk2gt_2,142,824). coords(wk2gt_2,99,63,824).  visible(wk2gt_2,824).
orientation(wk2gt_1,125,825). coords(wk2gt_1,115,53,825).  visible(wk2gt_1,825). orientation(wk2gt_2,142,825). coords(wk2gt_2,99,63,825).  visible(wk2gt_2,825).
orientation(wk2gt_1,125,826). coords(wk2gt_1,115,53,826).  visible(wk2gt_1,826). orientation(wk2gt_2,142,826). coords(wk2gt_2,99,63,826).  visible(wk2gt_2,826).
orientation(wk2gt_1,125,827). coords(wk2gt_1,115,53,827).  visible(wk2gt_1,827). orientation(wk2gt_2,142,827). coords(wk2gt_2,99,63,827).  visible(wk2gt_2,827).
orientation(wk2gt_1,125,828). coords(wk2gt_1,115,53,828).  visible(wk2gt_1,828). orientation(wk2gt_2,142,828). coords(wk2gt_2,99,63,828).  visible(wk2gt_2,828).
orientation(wk2gt_1,125,829). coords(wk2gt_1,115,53,829).  visible(wk2gt_1,829). orientation(wk2gt_2,142,829). coords(wk2gt_2,99,63,829).  visible(wk2gt_2,829).
happensAt(disappear(wk2gt_1),828). happensAt(disappear(wk2gt_2),828).
1{simpleEvent(T,wk2gt_1,active); simpleEvent(T,wk2gt_1,inactive); simpleEvent(T,wk2gt_1,walking); simpleEvent(T,wk2gt_1,running)}1 :- T=804..828.
1{simpleEvent(T,wk2gt_2,active); simpleEvent(T,wk2gt_2,inactive); simpleEvent(T,wk2gt_2,walking); simpleEvent(T,wk2gt_2,running)}1 :- T=804..828.
:- holdsAt(meeting(wk2gt_1,wk2gt_2),804). :- holdsAt(meeting(wk2gt_2,wk2gt_1),804).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),805). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),805).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),806). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),806).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),807). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),807).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),808). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),808).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),809). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),809).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),810). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),810).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),811). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),811).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),812). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),812).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),813). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),813).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),814). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),814).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),815). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),815).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),816). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),816).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),817). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),817).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),818). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),818).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),819). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),819).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),820). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),820).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),821). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),821).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),822). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),822).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),823). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),823).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),824). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),824).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),825). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),825).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),826). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),826).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),827). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),827).
:- not holdsAt(meeting(wk2gt_1,wk2gt_2),828). :- not holdsAt(meeting(wk2gt_2,wk2gt_1),828).
:- holdsAt(meeting(wk2gt_1,wk2gt_2),829). :- holdsAt(meeting(wk2gt_2,wk2gt_1),829).
:- simpleEvent(T,P,walking), T>804 , T<828.
:- simpleEvent(T,P,running), T>804 , T<828.

#show simpleEvent/3.