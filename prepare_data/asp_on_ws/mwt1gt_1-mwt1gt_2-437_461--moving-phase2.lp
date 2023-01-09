

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

	person(mwt1gt_1;mwt1gt_2).
orientation(mwt1gt_1,160,437). coords(mwt1gt_1,74,104,437).  visible(mwt1gt_1,437). orientation(mwt1gt_2,140,437). coords(mwt1gt_2,75,92,437).  visible(mwt1gt_2,437).
orientation(mwt1gt_1,160,438). coords(mwt1gt_1,74,104,438).  visible(mwt1gt_1,438). orientation(mwt1gt_2,140,438). coords(mwt1gt_2,75,92,438).  visible(mwt1gt_2,438).
orientation(mwt1gt_1,160,439). coords(mwt1gt_1,74,104,439).  visible(mwt1gt_1,439). orientation(mwt1gt_2,140,439). coords(mwt1gt_2,75,92,439).  visible(mwt1gt_2,439).
orientation(mwt1gt_1,160,440). coords(mwt1gt_1,74,104,440).  visible(mwt1gt_1,440). orientation(mwt1gt_2,140,440). coords(mwt1gt_2,75,92,440).  visible(mwt1gt_2,440).
orientation(mwt1gt_1,160,441). coords(mwt1gt_1,74,104,441).  visible(mwt1gt_1,441). orientation(mwt1gt_2,140,441). coords(mwt1gt_2,75,92,441).  visible(mwt1gt_2,441).
orientation(mwt1gt_1,160,442). coords(mwt1gt_1,74,104,442).  visible(mwt1gt_1,442). orientation(mwt1gt_2,140,442). coords(mwt1gt_2,75,92,442).  visible(mwt1gt_2,442).
orientation(mwt1gt_1,160,443). coords(mwt1gt_1,74,104,443).  visible(mwt1gt_1,443). orientation(mwt1gt_2,140,443). coords(mwt1gt_2,75,92,443).  visible(mwt1gt_2,443).
orientation(mwt1gt_1,160,444). coords(mwt1gt_1,74,104,444).  visible(mwt1gt_1,444). orientation(mwt1gt_2,140,444). coords(mwt1gt_2,75,92,444).  visible(mwt1gt_2,444).
orientation(mwt1gt_1,160,445). coords(mwt1gt_1,74,101,445).  visible(mwt1gt_1,445). orientation(mwt1gt_2,140,445). coords(mwt1gt_2,74,90,445).  visible(mwt1gt_2,445).
orientation(mwt1gt_1,160,446). coords(mwt1gt_1,74,101,446).  visible(mwt1gt_1,446). orientation(mwt1gt_2,140,446). coords(mwt1gt_2,74,90,446).  visible(mwt1gt_2,446).
orientation(mwt1gt_1,160,447). coords(mwt1gt_1,72,99,447).  visible(mwt1gt_1,447). orientation(mwt1gt_2,140,447). coords(mwt1gt_2,72,88,447).  visible(mwt1gt_2,447).
orientation(mwt1gt_1,160,448). coords(mwt1gt_1,72,99,448).  visible(mwt1gt_1,448). orientation(mwt1gt_2,140,448). coords(mwt1gt_2,72,88,448).  visible(mwt1gt_2,448).
orientation(mwt1gt_1,160,449). coords(mwt1gt_1,72,99,449).  visible(mwt1gt_1,449). orientation(mwt1gt_2,140,449). coords(mwt1gt_2,72,88,449).  visible(mwt1gt_2,449).
orientation(mwt1gt_1,160,450). coords(mwt1gt_1,72,99,450).  visible(mwt1gt_1,450). orientation(mwt1gt_2,140,450). coords(mwt1gt_2,72,88,450).  visible(mwt1gt_2,450).
orientation(mwt1gt_1,160,451). coords(mwt1gt_1,72,99,451).  visible(mwt1gt_1,451). orientation(mwt1gt_2,140,451). coords(mwt1gt_2,72,88,451).  visible(mwt1gt_2,451).
orientation(mwt1gt_1,160,452). coords(mwt1gt_1,72,99,452).  visible(mwt1gt_1,452). orientation(mwt1gt_2,140,452). coords(mwt1gt_2,72,88,452).  visible(mwt1gt_2,452).
orientation(mwt1gt_1,160,453). coords(mwt1gt_1,72,97,453).  visible(mwt1gt_1,453). orientation(mwt1gt_2,140,453). coords(mwt1gt_2,72,87,453).  visible(mwt1gt_2,453).
orientation(mwt1gt_1,160,454). coords(mwt1gt_1,69,97,454).  visible(mwt1gt_1,454). orientation(mwt1gt_2,140,454). coords(mwt1gt_2,72,87,454).  visible(mwt1gt_2,454).
orientation(mwt1gt_1,160,455). coords(mwt1gt_1,69,97,455).  visible(mwt1gt_1,455). orientation(mwt1gt_2,140,455). coords(mwt1gt_2,73,87,455).  visible(mwt1gt_2,455).
orientation(mwt1gt_1,160,456). coords(mwt1gt_1,69,97,456).  visible(mwt1gt_1,456). orientation(mwt1gt_2,140,456). coords(mwt1gt_2,73,87,456).  visible(mwt1gt_2,456).
orientation(mwt1gt_1,160,457). coords(mwt1gt_1,69,97,457).  visible(mwt1gt_1,457). orientation(mwt1gt_2,140,457). coords(mwt1gt_2,73,87,457).  visible(mwt1gt_2,457).
orientation(mwt1gt_1,160,458). coords(mwt1gt_1,69,97,458).  visible(mwt1gt_1,458). orientation(mwt1gt_2,140,458). coords(mwt1gt_2,73,87,458).  visible(mwt1gt_2,458).
orientation(mwt1gt_1,160,459). coords(mwt1gt_1,69,97,459).  visible(mwt1gt_1,459). orientation(mwt1gt_2,140,459). coords(mwt1gt_2,73,87,459).  visible(mwt1gt_2,459).
orientation(mwt1gt_1,160,460). coords(mwt1gt_1,69,97,460).  visible(mwt1gt_1,460). orientation(mwt1gt_2,140,460). coords(mwt1gt_2,73,87,460).  visible(mwt1gt_2,460).
orientation(mwt1gt_1,160,461). coords(mwt1gt_1,69,97,461).  visible(mwt1gt_1,461). orientation(mwt1gt_2,140,461). coords(mwt1gt_2,73,87,461).  visible(mwt1gt_2,461).
orientation(mwt1gt_1,160,462). coords(mwt1gt_1,69,97,462).  visible(mwt1gt_1,462). orientation(mwt1gt_2,140,462). coords(mwt1gt_2,73,87,462).  visible(mwt1gt_2,462).
happensAt(disappear(mwt1gt_1),461). happensAt(disappear(mwt1gt_2),461).
1{simpleEvent(T,mwt1gt_1,active); simpleEvent(T,mwt1gt_1,inactive); simpleEvent(T,mwt1gt_1,walking); simpleEvent(T,mwt1gt_1,running)}1 :- T=437..461.
1{simpleEvent(T,mwt1gt_2,active); simpleEvent(T,mwt1gt_2,inactive); simpleEvent(T,mwt1gt_2,walking); simpleEvent(T,mwt1gt_2,running)}1 :- T=437..461.
:- holdsAt(moving(mwt1gt_1,mwt1gt_2),437). :- holdsAt(moving(mwt1gt_2,mwt1gt_1),437).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),438). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),438).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),439). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),439).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),440). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),440).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),441). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),441).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),442). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),442).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),443). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),443).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),444). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),444).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),445). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),445).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),446). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),446).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),447). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),447).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),448). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),448).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),449). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),449).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),450). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),450).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),451). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),451).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),452). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),452).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),453). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),453).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),454). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),454).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),455). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),455).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),456). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),456).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),457). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),457).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),458). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),458).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),459). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),459).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),460). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),460).
:- not holdsAt(moving(mwt1gt_1,mwt1gt_2),461). :- not holdsAt(moving(mwt1gt_2,mwt1gt_1),461).
:- holdsAt(moving(mwt1gt_1,mwt1gt_2),462). :- holdsAt(moving(mwt1gt_2,mwt1gt_1),462).
:- simpleEvent(T,P,inactive), T>437 , T<461.
:- simpleEvent(T,P,running), T>437 , T<461.

#show simpleEvent/3.