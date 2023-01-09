

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
orientation(ms3ggt_0,157,428). coords(ms3ggt_0,118,145,428).  visible(ms3ggt_0,428). orientation(ms3ggt_2,159,428). coords(ms3ggt_2,117,177,428).  visible(ms3ggt_2,428).
orientation(ms3ggt_0,157,429). coords(ms3ggt_0,118,145,429).  visible(ms3ggt_0,429). orientation(ms3ggt_2,159,429). coords(ms3ggt_2,117,176,429).  visible(ms3ggt_2,429).
orientation(ms3ggt_0,157,430). coords(ms3ggt_0,116,145,430).  visible(ms3ggt_0,430). orientation(ms3ggt_2,155,430). coords(ms3ggt_2,116,176,430).  visible(ms3ggt_2,430).
orientation(ms3ggt_0,160,431). coords(ms3ggt_0,114,144,431).  visible(ms3ggt_0,431). orientation(ms3ggt_2,155,431). coords(ms3ggt_2,116,175,431).  visible(ms3ggt_2,431).
orientation(ms3ggt_0,160,432). coords(ms3ggt_0,112,144,432).  visible(ms3ggt_0,432). orientation(ms3ggt_2,155,432). coords(ms3ggt_2,116,174,432).  visible(ms3ggt_2,432).
orientation(ms3ggt_0,160,433). coords(ms3ggt_0,111,144,433).  visible(ms3ggt_0,433). orientation(ms3ggt_2,155,433). coords(ms3ggt_2,115,174,433).  visible(ms3ggt_2,433).
orientation(ms3ggt_0,160,434). coords(ms3ggt_0,109,144,434).  visible(ms3ggt_0,434). orientation(ms3ggt_2,155,434). coords(ms3ggt_2,115,173,434).  visible(ms3ggt_2,434).
orientation(ms3ggt_0,160,435). coords(ms3ggt_0,108,144,435).  visible(ms3ggt_0,435). orientation(ms3ggt_2,155,435). coords(ms3ggt_2,115,173,435).  visible(ms3ggt_2,435).
orientation(ms3ggt_0,160,436). coords(ms3ggt_0,107,145,436).  visible(ms3ggt_0,436). orientation(ms3ggt_2,155,436). coords(ms3ggt_2,114,171,436).  visible(ms3ggt_2,436).
orientation(ms3ggt_0,160,437). coords(ms3ggt_0,107,146,437).  visible(ms3ggt_0,437). orientation(ms3ggt_2,155,437). coords(ms3ggt_2,112,171,437).  visible(ms3ggt_2,437).
orientation(ms3ggt_0,160,438). coords(ms3ggt_0,107,146,438).  visible(ms3ggt_0,438). orientation(ms3ggt_2,155,438). coords(ms3ggt_2,111,170,438).  visible(ms3ggt_2,438).
orientation(ms3ggt_0,160,439). coords(ms3ggt_0,107,146,439).  visible(ms3ggt_0,439). orientation(ms3ggt_2,155,439). coords(ms3ggt_2,111,169,439).  visible(ms3ggt_2,439).
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
orientation(ms3ggt_0,166,453). coords(ms3ggt_0,85,150,453).  visible(ms3ggt_0,453). orientation(ms3ggt_2,155,453). coords(ms3ggt_2,106,160,453).  visible(ms3ggt_2,453).
happensAt(disappear(ms3ggt_0),452). happensAt(disappear(ms3ggt_2),452).
1{simpleEvent(T,ms3ggt_0,active); simpleEvent(T,ms3ggt_0,inactive); simpleEvent(T,ms3ggt_0,walking); simpleEvent(T,ms3ggt_0,running)}1 :- T=428..452.
1{simpleEvent(T,ms3ggt_2,active); simpleEvent(T,ms3ggt_2,inactive); simpleEvent(T,ms3ggt_2,walking); simpleEvent(T,ms3ggt_2,running)}1 :- T=428..452.
:- holdsAt(moving(ms3ggt_0,ms3ggt_2),428). :- holdsAt(moving(ms3ggt_2,ms3ggt_0),428).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),429). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),429).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),430). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),430).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),431). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),431).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),432). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),432).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),433). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),433).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),434). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),434).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),435). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),435).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),436). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),436).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),437). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),437).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),438). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),438).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),439). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),439).
:- not holdsAt(moving(ms3ggt_0,ms3ggt_2),440). :- not holdsAt(moving(ms3ggt_2,ms3ggt_0),440).
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
:- holdsAt(moving(ms3ggt_0,ms3ggt_2),453). :- holdsAt(moving(ms3ggt_2,ms3ggt_0),453).
:- simpleEvent(T,P,inactive), T>428 , T<452.
:- simpleEvent(T,P,running), T>428 , T<452.

#show simpleEvent/3.