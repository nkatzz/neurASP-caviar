

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
orientation(wk1gt_4,147,435). coords(wk1gt_4,55,170,435).  visible(wk1gt_4,435). orientation(wk1gt_5,162,435). coords(wk1gt_5,51,162,435).  visible(wk1gt_5,435).
orientation(wk1gt_4,147,436). coords(wk1gt_4,55,169,436).  visible(wk1gt_4,436). orientation(wk1gt_5,162,436). coords(wk1gt_5,52,160,436).  visible(wk1gt_5,436).
orientation(wk1gt_4,147,437). coords(wk1gt_4,56,169,437).  visible(wk1gt_4,437). orientation(wk1gt_5,162,437). coords(wk1gt_5,52,160,437).  visible(wk1gt_5,437).
orientation(wk1gt_4,147,438). coords(wk1gt_4,56,169,438).  visible(wk1gt_4,438). orientation(wk1gt_5,162,438). coords(wk1gt_5,52,160,438).  visible(wk1gt_5,438).
orientation(wk1gt_4,147,439). coords(wk1gt_4,56,169,439).  visible(wk1gt_4,439). orientation(wk1gt_5,162,439). coords(wk1gt_5,52,160,439).  visible(wk1gt_5,439).
orientation(wk1gt_4,147,440). coords(wk1gt_4,56,169,440).  visible(wk1gt_4,440). orientation(wk1gt_5,162,440). coords(wk1gt_5,52,160,440).  visible(wk1gt_5,440).
orientation(wk1gt_4,147,441). coords(wk1gt_4,56,169,441).  visible(wk1gt_4,441). orientation(wk1gt_5,162,441). coords(wk1gt_5,53,160,441).  visible(wk1gt_5,441).
orientation(wk1gt_4,147,442). coords(wk1gt_4,56,169,442).  visible(wk1gt_4,442). orientation(wk1gt_5,162,442). coords(wk1gt_5,54,160,442).  visible(wk1gt_5,442).
orientation(wk1gt_4,147,443). coords(wk1gt_4,56,169,443).  visible(wk1gt_4,443). orientation(wk1gt_5,162,443). coords(wk1gt_5,54,160,443).  visible(wk1gt_5,443).
orientation(wk1gt_4,147,444). coords(wk1gt_4,56,169,444).  visible(wk1gt_4,444). orientation(wk1gt_5,162,444). coords(wk1gt_5,54,160,444).  visible(wk1gt_5,444).
orientation(wk1gt_4,147,445). coords(wk1gt_4,56,169,445).  visible(wk1gt_4,445). orientation(wk1gt_5,162,445). coords(wk1gt_5,54,160,445).  visible(wk1gt_5,445).
orientation(wk1gt_4,147,446). coords(wk1gt_4,56,169,446).  visible(wk1gt_4,446). orientation(wk1gt_5,162,446). coords(wk1gt_5,54,159,446).  visible(wk1gt_5,446).
orientation(wk1gt_4,147,447). coords(wk1gt_4,56,169,447).  visible(wk1gt_4,447). orientation(wk1gt_5,162,447). coords(wk1gt_5,54,159,447).  visible(wk1gt_5,447).
orientation(wk1gt_4,147,448). coords(wk1gt_4,56,169,448).  visible(wk1gt_4,448). orientation(wk1gt_5,162,448). coords(wk1gt_5,54,159,448).  visible(wk1gt_5,448).
orientation(wk1gt_4,147,449). coords(wk1gt_4,56,169,449).  visible(wk1gt_4,449). orientation(wk1gt_5,162,449). coords(wk1gt_5,54,159,449).  visible(wk1gt_5,449).
orientation(wk1gt_4,147,450). coords(wk1gt_4,56,169,450).  visible(wk1gt_4,450). orientation(wk1gt_5,162,450). coords(wk1gt_5,54,158,450).  visible(wk1gt_5,450).
orientation(wk1gt_4,147,451). coords(wk1gt_4,56,169,451).  visible(wk1gt_4,451). orientation(wk1gt_5,162,451). coords(wk1gt_5,54,158,451).  visible(wk1gt_5,451).
orientation(wk1gt_4,147,452). coords(wk1gt_4,56,169,452).  visible(wk1gt_4,452). orientation(wk1gt_5,162,452). coords(wk1gt_5,54,158,452).  visible(wk1gt_5,452).
orientation(wk1gt_4,147,453). coords(wk1gt_4,56,169,453).  visible(wk1gt_4,453). orientation(wk1gt_5,162,453). coords(wk1gt_5,54,158,453).  visible(wk1gt_5,453).
orientation(wk1gt_4,147,454). coords(wk1gt_4,56,169,454).  visible(wk1gt_4,454). orientation(wk1gt_5,162,454). coords(wk1gt_5,54,158,454).  visible(wk1gt_5,454).
orientation(wk1gt_4,147,455). coords(wk1gt_4,56,169,455).  visible(wk1gt_4,455). orientation(wk1gt_5,162,455). coords(wk1gt_5,54,158,455).  visible(wk1gt_5,455).
orientation(wk1gt_4,147,456). coords(wk1gt_4,56,169,456).  visible(wk1gt_4,456). orientation(wk1gt_5,162,456). coords(wk1gt_5,54,158,456).  visible(wk1gt_5,456).
orientation(wk1gt_4,147,457). coords(wk1gt_4,56,169,457).  visible(wk1gt_4,457). orientation(wk1gt_5,162,457). coords(wk1gt_5,54,156,457).  visible(wk1gt_5,457).
orientation(wk1gt_4,147,458). coords(wk1gt_4,56,169,458).  visible(wk1gt_4,458). orientation(wk1gt_5,162,458). coords(wk1gt_5,55,156,458).  visible(wk1gt_5,458).
orientation(wk1gt_4,147,459). coords(wk1gt_4,56,169,459).  visible(wk1gt_4,459). orientation(wk1gt_5,162,459). coords(wk1gt_5,55,156,459).  visible(wk1gt_5,459).
orientation(wk1gt_4,147,460). coords(wk1gt_4,56,169,460).  visible(wk1gt_4,460). orientation(wk1gt_5,162,460). coords(wk1gt_5,55,156,460).  visible(wk1gt_5,460).
happensAt(disappear(wk1gt_4),459). happensAt(disappear(wk1gt_5),459).
1{simpleEvent(T,wk1gt_4,active); simpleEvent(T,wk1gt_4,inactive); simpleEvent(T,wk1gt_4,walking); simpleEvent(T,wk1gt_4,running)}1 :- T=435..459.
1{simpleEvent(T,wk1gt_5,active); simpleEvent(T,wk1gt_5,inactive); simpleEvent(T,wk1gt_5,walking); simpleEvent(T,wk1gt_5,running)}1 :- T=435..459.
:- holdsAt(meeting(wk1gt_4,wk1gt_5),435). :- holdsAt(meeting(wk1gt_5,wk1gt_4),435).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),436). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),436).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),437). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),437).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),438). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),438).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),439). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),439).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),440). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),440).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),441). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),441).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),442). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),442).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),443). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),443).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),444). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),444).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),445). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),445).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),446). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),446).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),447). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),447).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),448). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),448).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),449). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),449).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),450). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),450).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),451). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),451).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),452). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),452).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),453). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),453).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),454). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),454).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),455). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),455).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),456). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),456).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),457). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),457).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),458). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),458).
:- not holdsAt(meeting(wk1gt_4,wk1gt_5),459). :- not holdsAt(meeting(wk1gt_5,wk1gt_4),459).
:- holdsAt(meeting(wk1gt_4,wk1gt_5),460). :- holdsAt(meeting(wk1gt_5,wk1gt_4),460).
:- simpleEvent(T,P,walking), T>435 , T<459.
:- simpleEvent(T,P,running), T>435 , T<459.

#show simpleEvent/3.