

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
orientation(mwt2gt_1,153,802). coords(mwt2gt_1,30,122,802).  visible(mwt2gt_1,802). orientation(mwt2gt_2,152,802). coords(mwt2gt_2,28,125,802).  visible(mwt2gt_2,802).
orientation(mwt2gt_1,153,803). coords(mwt2gt_1,30,122,803).  visible(mwt2gt_1,803). orientation(mwt2gt_2,152,803). coords(mwt2gt_2,28,125,803).  visible(mwt2gt_2,803).
orientation(mwt2gt_1,153,804). coords(mwt2gt_1,30,122,804).  visible(mwt2gt_1,804). orientation(mwt2gt_2,152,804). coords(mwt2gt_2,28,125,804).  visible(mwt2gt_2,804).
orientation(mwt2gt_1,153,805). coords(mwt2gt_1,30,122,805).  visible(mwt2gt_1,805). orientation(mwt2gt_2,152,805). coords(mwt2gt_2,28,125,805).  visible(mwt2gt_2,805).
orientation(mwt2gt_1,153,806). coords(mwt2gt_1,29,124,806).  visible(mwt2gt_1,806). orientation(mwt2gt_2,152,806). coords(mwt2gt_2,28,125,806).  visible(mwt2gt_2,806).
orientation(mwt2gt_1,153,807). coords(mwt2gt_1,29,124,807).  visible(mwt2gt_1,807). orientation(mwt2gt_2,152,807). coords(mwt2gt_2,28,127,807).  visible(mwt2gt_2,807).
orientation(mwt2gt_1,153,808). coords(mwt2gt_1,29,124,808).  visible(mwt2gt_1,808). orientation(mwt2gt_2,152,808). coords(mwt2gt_2,28,127,808).  visible(mwt2gt_2,808).
orientation(mwt2gt_1,153,809). coords(mwt2gt_1,30,122,809).  visible(mwt2gt_1,809). orientation(mwt2gt_2,152,809). coords(mwt2gt_2,28,128,809).  visible(mwt2gt_2,809).
orientation(mwt2gt_1,153,810). coords(mwt2gt_1,30,122,810).  visible(mwt2gt_1,810). orientation(mwt2gt_2,152,810). coords(mwt2gt_2,28,128,810).  visible(mwt2gt_2,810).
orientation(mwt2gt_1,153,811). coords(mwt2gt_1,30,122,811).  visible(mwt2gt_1,811). orientation(mwt2gt_2,152,811). coords(mwt2gt_2,28,128,811).  visible(mwt2gt_2,811).
orientation(mwt2gt_1,153,812). coords(mwt2gt_1,30,122,812).  visible(mwt2gt_1,812). orientation(mwt2gt_2,152,812). coords(mwt2gt_2,28,128,812).  visible(mwt2gt_2,812).
orientation(mwt2gt_1,153,813). coords(mwt2gt_1,28,122,813).  visible(mwt2gt_1,813). orientation(mwt2gt_2,152,813). coords(mwt2gt_2,28,128,813).  visible(mwt2gt_2,813).
orientation(mwt2gt_1,153,814). coords(mwt2gt_1,28,122,814).  visible(mwt2gt_1,814). orientation(mwt2gt_2,152,814). coords(mwt2gt_2,28,128,814).  visible(mwt2gt_2,814).
orientation(mwt2gt_1,153,815). coords(mwt2gt_1,28,122,815).  visible(mwt2gt_1,815). orientation(mwt2gt_2,152,815). coords(mwt2gt_2,28,129,815).  visible(mwt2gt_2,815).
orientation(mwt2gt_1,153,816). coords(mwt2gt_1,28,122,816).  visible(mwt2gt_1,816). orientation(mwt2gt_2,152,816). coords(mwt2gt_2,28,129,816).  visible(mwt2gt_2,816).
orientation(mwt2gt_1,153,817). coords(mwt2gt_1,28,122,817).  visible(mwt2gt_1,817). orientation(mwt2gt_2,152,817). coords(mwt2gt_2,27,129,817).  visible(mwt2gt_2,817).
orientation(mwt2gt_1,153,818). coords(mwt2gt_1,28,122,818).  visible(mwt2gt_1,818). orientation(mwt2gt_2,152,818). coords(mwt2gt_2,27,129,818).  visible(mwt2gt_2,818).
orientation(mwt2gt_1,153,819). coords(mwt2gt_1,28,122,819).  visible(mwt2gt_1,819). orientation(mwt2gt_2,152,819). coords(mwt2gt_2,27,130,819).  visible(mwt2gt_2,819).
orientation(mwt2gt_1,153,820). coords(mwt2gt_1,28,122,820).  visible(mwt2gt_1,820). orientation(mwt2gt_2,152,820). coords(mwt2gt_2,27,130,820).  visible(mwt2gt_2,820).
orientation(mwt2gt_1,153,821). coords(mwt2gt_1,28,123,821).  visible(mwt2gt_1,821). orientation(mwt2gt_2,152,821). coords(mwt2gt_2,27,130,821).  visible(mwt2gt_2,821).
orientation(mwt2gt_1,153,822). coords(mwt2gt_1,28,123,822).  visible(mwt2gt_1,822). orientation(mwt2gt_2,152,822). coords(mwt2gt_2,27,130,822).  visible(mwt2gt_2,822).
orientation(mwt2gt_1,153,823). coords(mwt2gt_1,28,123,823).  visible(mwt2gt_1,823). orientation(mwt2gt_2,152,823). coords(mwt2gt_2,27,131,823).  visible(mwt2gt_2,823).
orientation(mwt2gt_1,153,824). coords(mwt2gt_1,28,123,824).  visible(mwt2gt_1,824). orientation(mwt2gt_2,152,824). coords(mwt2gt_2,27,131,824).  visible(mwt2gt_2,824).
orientation(mwt2gt_1,153,825). coords(mwt2gt_1,28,123,825).  visible(mwt2gt_1,825). orientation(mwt2gt_2,152,825). coords(mwt2gt_2,27,131,825).  visible(mwt2gt_2,825).
orientation(mwt2gt_1,153,826). coords(mwt2gt_1,29,123,826).  happensAt(disappear(mwt2gt_1),826). orientation(mwt2gt_2,152,826). coords(mwt2gt_2,27,131,826).  happensAt(disappear(mwt2gt_2),826).
orientation(mwt2gt_1,153,827). coords(mwt2gt_1,29,123,827).  happensAt(disappear(mwt2gt_1),827). orientation(mwt2gt_2,152,827). coords(mwt2gt_2,27,131,827).  happensAt(disappear(mwt2gt_2),827).
happensAt(disappear(mwt2gt_1),826). happensAt(disappear(mwt2gt_2),826).
1{simpleEvent(T,mwt2gt_1,active); simpleEvent(T,mwt2gt_1,inactive); simpleEvent(T,mwt2gt_1,walking); simpleEvent(T,mwt2gt_1,running)}1 :- T=802..826.
1{simpleEvent(T,mwt2gt_2,active); simpleEvent(T,mwt2gt_2,inactive); simpleEvent(T,mwt2gt_2,walking); simpleEvent(T,mwt2gt_2,running)}1 :- T=802..826.
:- holdsAt(meeting(mwt2gt_1,mwt2gt_2),802). :- holdsAt(meeting(mwt2gt_2,mwt2gt_1),802).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),803). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),803).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),804). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),804).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),805). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),805).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),806). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),806).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),807). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),807).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),808). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),808).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),809). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),809).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),810). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),810).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),811). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),811).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),812). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),812).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),813). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),813).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),814). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),814).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),815). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),815).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),816). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),816).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),817). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),817).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),818). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),818).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),819). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),819).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),820). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),820).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),821). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),821).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),822). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),822).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),823). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),823).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),824). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),824).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),825). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),825).
:- not holdsAt(meeting(mwt2gt_1,mwt2gt_2),826). :- not holdsAt(meeting(mwt2gt_2,mwt2gt_1),826).
:- holdsAt(meeting(mwt2gt_1,mwt2gt_2),827). :- holdsAt(meeting(mwt2gt_2,mwt2gt_1),827).
:- simpleEvent(T,P,walking), T>802 , T<826.
:- simpleEvent(T,P,running), T>802 , T<826.

#show simpleEvent/3.