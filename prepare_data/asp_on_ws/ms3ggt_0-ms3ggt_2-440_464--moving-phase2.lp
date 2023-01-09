

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

	person(ms3ggt_0;ms3ggt_2).
orientation(ms3ggt_0,168,440). coords(ms3ggt_0,106,146,440).  visible(ms3ggt_0,440). orientation(ms3ggt_2,155,440). coords(ms3ggt_2,110,169,440).  visible(ms3ggt_2,440).
orientation(ms3ggt_0,168,441). coords(ms3ggt_0,105,147,441).  visible(ms3ggt_0,441). orientation(ms3ggt_2,155,441). coords(ms3ggt_2,109,167,441).  visible(ms3ggt_2,441).
orientation(ms3ggt_0,168,442). coords(ms3ggt_0,105,147,442).  visible(ms3ggt_0,442). orientation(ms3ggt_2,155,442). coords(ms3ggt_2,109,167,442).  visible(ms3ggt_2,442).
orientation(ms3ggt_0,170,443). coords(ms3ggt_0,103,147,443).  visible(ms3ggt_0,443). orientation(ms3ggt_2,155,443). coords(ms3ggt_2,109,167,443).  visible(ms3ggt_2,443).
orientation(ms3ggt_0,171,444). coords(ms3ggt_0,101,148,444).  visible(ms3ggt_0,444). orientation(ms3ggt_2,155,444). coords(ms3ggt_2,109,164,444).  visible(ms3ggt_2,444).
orientation(ms3ggt_0,171,445). coords(ms3ggt_0,100,148,445).  visible(ms3ggt_0,445). orientation(ms3ggt_2,155,445). coords(ms3ggt_2,109,164,445).  visible(ms3ggt_2,445).
orientation(ms3ggt_0,171,446). coords(ms3ggt_0,98,148,446).  visible(ms3ggt_0,446). orientation(ms3ggt_2,155,446). coords(ms3ggt_2,108,164,446).  visible(ms3ggt_2,446).
orientation(ms3ggt_0,171,447). coords(ms3ggt_0,97,148,447).  visible(ms3ggt_0,447). orientation(ms3ggt_2,155,447). coords(ms3ggt_2,108,164,447).  visible(ms3ggt_2,447).
orientation(ms3ggt_0,171,448). coords(ms3ggt_0,96,148,448).  visible(ms3ggt_0,448). orientation(ms3ggt_2,155,448). coords(ms3ggt_2,108,163,448).  visible(ms3ggt_2,448).
orientation(ms3ggt_0,171,449). coords(ms3ggt_0,96,148,449).  visible(ms3ggt_0,449). orientation(ms3ggt_2,155,449). coords(ms3ggt_2,108,162,449).  visible(ms3ggt_2,449).
orientation(ms3ggt_0,166,450). coords(ms3ggt_0,87,149,450).  visible(ms3ggt_0,450). orientation(ms3ggt_2,155,450). coords(ms3ggt_2,108,161,450).  visible(ms3ggt_2,450).
orientation(ms3ggt_0,166,451). coords(ms3ggt_0,87,149,451).  visible(ms3ggt_0,451). orientation(ms3ggt_2,155,451). coords(ms3ggt_2,107,161,451).  visible(ms3ggt_2,451).
orientation(ms3ggt_0,166,452). coords(ms3ggt_0,85,150,452).  visible(ms3ggt_0,452). orientation(ms3ggt_2,155,452). coords(ms3ggt_2,106,160,452).  visible(ms3ggt_2,452).
orientation(ms3ggt_0,166,453). coords(ms3ggt_0,85,150,453).  visible(ms3ggt_0,453). orientation(ms3ggt_2,156,453). coords(ms3ggt_2,106,159,453).  visible(ms3ggt_2,453).
orientation(ms3ggt_0,166,454). coords(ms3ggt_0,83,151,454).  visible(ms3ggt_0,454). orientation(ms3ggt_2,156,454). coords(ms3ggt_2,105,159,454).  visible(ms3ggt_2,454).
orientation(ms3ggt_0,166,455). coords(ms3ggt_0,82,151,455).  visible(ms3ggt_0,455). orientation(ms3ggt_2,156,455). coords(ms3ggt_2,105,158,455).  visible(ms3ggt_2,455).
orientation(ms3ggt_0,166,456). coords(ms3ggt_0,82,151,456).  visible(ms3ggt_0,456). orientation(ms3ggt_2,156,456). coords(ms3ggt_2,104,157,456).  visible(ms3ggt_2,456).
orientation(ms3ggt_0,166,457). coords(ms3ggt_0,78,152,457).  visible(ms3ggt_0,457). orientation(ms3ggt_2,156,457). coords(ms3ggt_2,104,158,457).  visible(ms3ggt_2,457).
orientation(ms3ggt_0,166,458). coords(ms3ggt_0,77,152,458).  visible(ms3ggt_0,458). orientation(ms3ggt_2,156,458). coords(ms3ggt_2,103,156,458).  visible(ms3ggt_2,458).
orientation(ms3ggt_0,166,459). coords(ms3ggt_0,86,156,459).  visible(ms3ggt_0,459). orientation(ms3ggt_2,156,459). coords(ms3ggt_2,103,156,459).  visible(ms3ggt_2,459).
orientation(ms3ggt_0,166,460). coords(ms3ggt_0,86,157,460).  visible(ms3ggt_0,460). orientation(ms3ggt_2,156,460). coords(ms3ggt_2,103,155,460).  visible(ms3ggt_2,460).
orientation(ms3ggt_0,166,461). coords(ms3ggt_0,87,158,461).  visible(ms3ggt_0,461). orientation(ms3ggt_2,156,461). coords(ms3ggt_2,103,155,461).  visible(ms3ggt_2,461).
orientation(ms3ggt_0,166,462). coords(ms3ggt_0,86,158,462).  visible(ms3ggt_0,462). orientation(ms3ggt_2,151,462). coords(ms3ggt_2,103,154,462).  visible(ms3ggt_2,462).
orientation(ms3ggt_0,166,463). coords(ms3ggt_0,85,161,463).  visible(ms3ggt_0,463). orientation(ms3ggt_2,151,463). coords(ms3ggt_2,103,153,463).  visible(ms3ggt_2,463).
orientation(ms3ggt_0,155,464). coords(ms3ggt_0,84,162,464).  visible(ms3ggt_0,464). orientation(ms3ggt_2,151,464). coords(ms3ggt_2,102,152,464).  visible(ms3ggt_2,464).
orientation(ms3ggt_0,155,465). coords(ms3ggt_0,84,162,465).  visible(ms3ggt_0,465). orientation(ms3ggt_2,151,465). coords(ms3ggt_2,102,152,465).  visible(ms3ggt_2,465).
happensAt(disappear(ms3ggt_0),464). happensAt(disappear(ms3ggt_2),464).
1{simpleEvent(T,ms3ggt_0,active); simpleEvent(T,ms3ggt_0,inactive); simpleEvent(T,ms3ggt_0,walking); simpleEvent(T,ms3ggt_0,running)}1 :- T=440..464.
1{simpleEvent(T,ms3ggt_2,active); simpleEvent(T,ms3ggt_2,inactive); simpleEvent(T,ms3ggt_2,walking); simpleEvent(T,ms3ggt_2,running)}1 :- T=440..464.
:- holdsAt(moving(ms3ggt_0,ms3ggt_2),440). :- holdsAt(moving(ms3ggt_2,ms3ggt_0),440).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),441). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),441).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),442). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),442).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),443). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),443).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),444). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),444).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),445). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),445).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),446). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),446).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),447). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),447).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),448). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),448).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),449). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),449).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),450). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),450).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),451). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),451).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),452). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),452).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),453). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),453).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),454). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),454).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),455). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),455).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),456). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),456).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),457). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),457).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),458). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),458).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),459). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),459).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),460). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),460).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),461). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),461).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),462). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),462).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),463). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),463).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),464). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),464).
:- holdsAt(moving(ms3ggt_0,ms3ggt_2),465). :- holdsAt(moving(ms3ggt_2,ms3ggt_0),465).
:- simpleEvent(T,P,inactive), T>440 , T<464.
:- simpleEvent(T,P,running), T>440 , T<464.

#show simpleEvent/3.