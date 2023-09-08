

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
orientation(mwt2gt_1,153,433). coords(mwt2gt_1,87,92,433).  visible(mwt2gt_1,433). orientation(mwt2gt_2,150,433). coords(mwt2gt_2,84,122,433).  visible(mwt2gt_2,433).
orientation(mwt2gt_1,153,434). coords(mwt2gt_1,86,92,434).  visible(mwt2gt_1,434). orientation(mwt2gt_2,150,434). coords(mwt2gt_2,83,123,434).  visible(mwt2gt_2,434).
orientation(mwt2gt_1,153,435). coords(mwt2gt_1,86,93,435).  visible(mwt2gt_1,435). orientation(mwt2gt_2,150,435). coords(mwt2gt_2,83,123,435).  visible(mwt2gt_2,435).
orientation(mwt2gt_1,153,436). coords(mwt2gt_1,85,94,436).  visible(mwt2gt_1,436). orientation(mwt2gt_2,150,436). coords(mwt2gt_2,82,123,436).  visible(mwt2gt_2,436).
orientation(mwt2gt_1,153,437). coords(mwt2gt_1,85,95,437).  visible(mwt2gt_1,437). orientation(mwt2gt_2,150,437). coords(mwt2gt_2,82,123,437).  visible(mwt2gt_2,437).
orientation(mwt2gt_1,153,438). coords(mwt2gt_1,85,97,438).  visible(mwt2gt_1,438). orientation(mwt2gt_2,150,438). coords(mwt2gt_2,81,123,438).  visible(mwt2gt_2,438).
orientation(mwt2gt_1,150,439). coords(mwt2gt_1,84,98,439).  visible(mwt2gt_1,439). orientation(mwt2gt_2,150,439). coords(mwt2gt_2,81,123,439).  visible(mwt2gt_2,439).
orientation(mwt2gt_1,150,440). coords(mwt2gt_1,84,99,440).  visible(mwt2gt_1,440). orientation(mwt2gt_2,150,440). coords(mwt2gt_2,81,124,440).  visible(mwt2gt_2,440).
orientation(mwt2gt_1,155,441). coords(mwt2gt_1,84,101,441).  visible(mwt2gt_1,441). orientation(mwt2gt_2,150,441). coords(mwt2gt_2,81,125,441).  visible(mwt2gt_2,441).
orientation(mwt2gt_1,155,442). coords(mwt2gt_1,83,101,442).  visible(mwt2gt_1,442). orientation(mwt2gt_2,150,442). coords(mwt2gt_2,80,126,442).  visible(mwt2gt_2,442).
orientation(mwt2gt_1,155,443). coords(mwt2gt_1,83,102,443).  visible(mwt2gt_1,443). orientation(mwt2gt_2,150,443). coords(mwt2gt_2,80,127,443).  visible(mwt2gt_2,443).
orientation(mwt2gt_1,155,444). coords(mwt2gt_1,82,102,444).  visible(mwt2gt_1,444). orientation(mwt2gt_2,150,444). coords(mwt2gt_2,80,128,444).  visible(mwt2gt_2,444).
orientation(mwt2gt_1,155,445). coords(mwt2gt_1,81,103,445).  visible(mwt2gt_1,445). orientation(mwt2gt_2,150,445). coords(mwt2gt_2,80,128,445).  visible(mwt2gt_2,445).
orientation(mwt2gt_1,155,446). coords(mwt2gt_1,80,104,446).  visible(mwt2gt_1,446). orientation(mwt2gt_2,150,446). coords(mwt2gt_2,79,129,446).  visible(mwt2gt_2,446).
orientation(mwt2gt_1,155,447). coords(mwt2gt_1,78,104,447).  visible(mwt2gt_1,447). orientation(mwt2gt_2,150,447). coords(mwt2gt_2,78,129,447).  visible(mwt2gt_2,447).
orientation(mwt2gt_1,155,448). coords(mwt2gt_1,78,104,448).  visible(mwt2gt_1,448). orientation(mwt2gt_2,150,448). coords(mwt2gt_2,78,130,448).  visible(mwt2gt_2,448).
orientation(mwt2gt_1,155,449). coords(mwt2gt_1,78,105,449).  visible(mwt2gt_1,449). orientation(mwt2gt_2,150,449). coords(mwt2gt_2,77,130,449).  visible(mwt2gt_2,449).
orientation(mwt2gt_1,155,450). coords(mwt2gt_1,77,106,450).  visible(mwt2gt_1,450). orientation(mwt2gt_2,150,450). coords(mwt2gt_2,75,130,450).  visible(mwt2gt_2,450).
orientation(mwt2gt_1,155,451). coords(mwt2gt_1,77,109,451).  visible(mwt2gt_1,451). orientation(mwt2gt_2,157,451). coords(mwt2gt_2,74,131,451).  visible(mwt2gt_2,451).
orientation(mwt2gt_1,155,452). coords(mwt2gt_1,77,110,452).  visible(mwt2gt_1,452). orientation(mwt2gt_2,157,452). coords(mwt2gt_2,74,131,452).  visible(mwt2gt_2,452).
orientation(mwt2gt_1,155,453). coords(mwt2gt_1,77,111,453).  visible(mwt2gt_1,453). orientation(mwt2gt_2,157,453). coords(mwt2gt_2,73,131,453).  visible(mwt2gt_2,453).
orientation(mwt2gt_1,155,454). coords(mwt2gt_1,77,113,454).  visible(mwt2gt_1,454). orientation(mwt2gt_2,157,454). coords(mwt2gt_2,73,132,454).  visible(mwt2gt_2,454).
orientation(mwt2gt_1,155,455). coords(mwt2gt_1,76,114,455).  visible(mwt2gt_1,455). orientation(mwt2gt_2,157,455). coords(mwt2gt_2,72,133,455).  visible(mwt2gt_2,455).
orientation(mwt2gt_1,155,456). coords(mwt2gt_1,75,114,456).  visible(mwt2gt_1,456). orientation(mwt2gt_2,157,456). coords(mwt2gt_2,72,135,456).  visible(mwt2gt_2,456).
orientation(mwt2gt_1,155,457). coords(mwt2gt_1,74,115,457).  visible(mwt2gt_1,457). orientation(mwt2gt_2,157,457). coords(mwt2gt_2,72,135,457).  visible(mwt2gt_2,457).
orientation(mwt2gt_1,155,458). coords(mwt2gt_1,74,115,458).  visible(mwt2gt_1,458). orientation(mwt2gt_2,157,458). coords(mwt2gt_2,72,135,458).  visible(mwt2gt_2,458).
happensAt(disappear(mwt2gt_1),457). happensAt(disappear(mwt2gt_2),457).
1{simpleEvent(T,mwt2gt_1,active); simpleEvent(T,mwt2gt_1,inactive); simpleEvent(T,mwt2gt_1,walking); simpleEvent(T,mwt2gt_1,running)}1 :- T=433..457.
1{simpleEvent(T,mwt2gt_2,active); simpleEvent(T,mwt2gt_2,inactive); simpleEvent(T,mwt2gt_2,walking); simpleEvent(T,mwt2gt_2,running)}1 :- T=433..457.
:- holdsAt(moving(mwt2gt_1,mwt2gt_2),433). :- holdsAt(moving(mwt2gt_2,mwt2gt_1),433).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),434). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),434).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),435). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),435).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),436). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),436).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),437). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),437).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),438). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),438).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),439). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),439).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),440). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),440).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),441). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),441).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),442). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),442).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),443). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),443).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),444). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),444).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),445). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),445).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),446). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),446).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),447). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),447).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),448). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),448).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),449). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),449).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),450). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),450).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),451). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),451).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),452). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),452).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),453). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),453).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),454). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),454).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),455). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),455).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),456). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),456).
:- not holdsAt(moving(mwt2gt_1,mwt2gt_2),457). :- not holdsAt(moving(mwt2gt_2,mwt2gt_1),457).
:- holdsAt(moving(mwt2gt_1,mwt2gt_2),458). :- holdsAt(moving(mwt2gt_2,mwt2gt_1),458).
:- simpleEvent(T,P,inactive), T>433 , T<457.
:- simpleEvent(T,P,running), T>433 , T<457.

#show simpleEvent/3.