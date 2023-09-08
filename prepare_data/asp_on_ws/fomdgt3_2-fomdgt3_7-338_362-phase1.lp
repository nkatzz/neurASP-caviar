
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


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


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
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(fomdgt3_2,30,338). coords(fomdgt3_2,263,81,338).  visible(fomdgt3_2,338). happensAt(walking(fomdgt3_2),338). orientation(fomdgt3_7,134,338). coords(fomdgt3_7,83,67,338).  happensAt(appear(fomdgt3_7),338). happensAt(walking(fomdgt3_7),338).
orientation(fomdgt3_2,30,339). coords(fomdgt3_2,263,81,339).  visible(fomdgt3_2,339). happensAt(walking(fomdgt3_2),339). orientation(fomdgt3_7,134,339). coords(fomdgt3_7,83,67,339).  visible(fomdgt3_7,339). happensAt(walking(fomdgt3_7),339).
orientation(fomdgt3_2,30,340). coords(fomdgt3_2,263,81,340).  visible(fomdgt3_2,340). happensAt(walking(fomdgt3_2),340). orientation(fomdgt3_7,134,340). coords(fomdgt3_7,83,67,340).  visible(fomdgt3_7,340). happensAt(walking(fomdgt3_7),340).
orientation(fomdgt3_2,30,341). coords(fomdgt3_2,263,81,341).  visible(fomdgt3_2,341). happensAt(walking(fomdgt3_2),341). orientation(fomdgt3_7,134,341). coords(fomdgt3_7,83,67,341).  visible(fomdgt3_7,341). happensAt(walking(fomdgt3_7),341).
orientation(fomdgt3_2,30,342). coords(fomdgt3_2,274,81,342).  visible(fomdgt3_2,342). happensAt(walking(fomdgt3_2),342). orientation(fomdgt3_7,134,342). coords(fomdgt3_7,82,67,342).  visible(fomdgt3_7,342). happensAt(walking(fomdgt3_7),342).
orientation(fomdgt3_2,30,343). coords(fomdgt3_2,274,81,343).  visible(fomdgt3_2,343). happensAt(walking(fomdgt3_2),343). orientation(fomdgt3_7,134,343). coords(fomdgt3_7,82,67,343).  visible(fomdgt3_7,343). happensAt(walking(fomdgt3_7),343).
orientation(fomdgt3_2,30,344). coords(fomdgt3_2,274,81,344).  visible(fomdgt3_2,344). happensAt(walking(fomdgt3_2),344). orientation(fomdgt3_7,134,344). coords(fomdgt3_7,82,67,344).  visible(fomdgt3_7,344). happensAt(walking(fomdgt3_7),344).
orientation(fomdgt3_2,30,345). coords(fomdgt3_2,274,81,345).  visible(fomdgt3_2,345). happensAt(walking(fomdgt3_2),345). orientation(fomdgt3_7,134,345). coords(fomdgt3_7,82,67,345).  visible(fomdgt3_7,345). happensAt(walking(fomdgt3_7),345).
orientation(fomdgt3_2,30,346). coords(fomdgt3_2,274,80,346).  visible(fomdgt3_2,346). happensAt(walking(fomdgt3_2),346). orientation(fomdgt3_7,134,346). coords(fomdgt3_7,81,69,346).  visible(fomdgt3_7,346). happensAt(walking(fomdgt3_7),346).
orientation(fomdgt3_2,30,347). coords(fomdgt3_2,273,80,347).  visible(fomdgt3_2,347). happensAt(walking(fomdgt3_2),347). orientation(fomdgt3_7,134,347). coords(fomdgt3_7,81,69,347).  visible(fomdgt3_7,347). happensAt(walking(fomdgt3_7),347).
orientation(fomdgt3_2,30,348). coords(fomdgt3_2,273,80,348).  visible(fomdgt3_2,348). happensAt(walking(fomdgt3_2),348). orientation(fomdgt3_7,134,348). coords(fomdgt3_7,79,69,348).  visible(fomdgt3_7,348). happensAt(walking(fomdgt3_7),348).
orientation(fomdgt3_2,30,349). coords(fomdgt3_2,273,80,349).  visible(fomdgt3_2,349). happensAt(walking(fomdgt3_2),349). orientation(fomdgt3_7,134,349). coords(fomdgt3_7,78,70,349).  visible(fomdgt3_7,349). happensAt(walking(fomdgt3_7),349).
orientation(fomdgt3_2,30,350). coords(fomdgt3_2,271,80,350).  visible(fomdgt3_2,350). happensAt(walking(fomdgt3_2),350). orientation(fomdgt3_7,134,350). coords(fomdgt3_7,78,71,350).  visible(fomdgt3_7,350). happensAt(walking(fomdgt3_7),350).
orientation(fomdgt3_2,30,351). coords(fomdgt3_2,271,80,351).  visible(fomdgt3_2,351). happensAt(walking(fomdgt3_2),351). orientation(fomdgt3_7,134,351). coords(fomdgt3_7,78,71,351).  visible(fomdgt3_7,351). happensAt(walking(fomdgt3_7),351).
orientation(fomdgt3_2,30,352). coords(fomdgt3_2,270,81,352).  visible(fomdgt3_2,352). happensAt(walking(fomdgt3_2),352). orientation(fomdgt3_7,134,352). coords(fomdgt3_7,78,71,352).  visible(fomdgt3_7,352). happensAt(walking(fomdgt3_7),352).
orientation(fomdgt3_2,42,353). coords(fomdgt3_2,270,81,353).  visible(fomdgt3_2,353). happensAt(walking(fomdgt3_2),353). orientation(fomdgt3_7,134,353). coords(fomdgt3_7,78,75,353).  visible(fomdgt3_7,353). happensAt(walking(fomdgt3_7),353).
orientation(fomdgt3_2,42,354). coords(fomdgt3_2,268,81,354).  visible(fomdgt3_2,354). happensAt(walking(fomdgt3_2),354). orientation(fomdgt3_7,134,354). coords(fomdgt3_7,78,75,354).  visible(fomdgt3_7,354). happensAt(walking(fomdgt3_7),354).
orientation(fomdgt3_2,42,355). coords(fomdgt3_2,267,82,355).  visible(fomdgt3_2,355). happensAt(walking(fomdgt3_2),355). orientation(fomdgt3_7,134,355). coords(fomdgt3_7,78,75,355).  visible(fomdgt3_7,355). happensAt(walking(fomdgt3_7),355).
orientation(fomdgt3_2,42,356). coords(fomdgt3_2,265,84,356).  visible(fomdgt3_2,356). happensAt(walking(fomdgt3_2),356). orientation(fomdgt3_7,134,356). coords(fomdgt3_7,78,76,356).  visible(fomdgt3_7,356). happensAt(walking(fomdgt3_7),356).
orientation(fomdgt3_2,42,357). coords(fomdgt3_2,264,86,357).  visible(fomdgt3_2,357). happensAt(walking(fomdgt3_2),357). orientation(fomdgt3_7,134,357). coords(fomdgt3_7,78,76,357).  visible(fomdgt3_7,357). happensAt(walking(fomdgt3_7),357).
orientation(fomdgt3_2,42,358). coords(fomdgt3_2,264,88,358).  visible(fomdgt3_2,358). happensAt(walking(fomdgt3_2),358). orientation(fomdgt3_7,134,358). coords(fomdgt3_7,78,76,358).  visible(fomdgt3_7,358). happensAt(walking(fomdgt3_7),358).
orientation(fomdgt3_2,42,359). coords(fomdgt3_2,264,91,359).  visible(fomdgt3_2,359). happensAt(walking(fomdgt3_2),359). orientation(fomdgt3_7,134,359). coords(fomdgt3_7,78,76,359).  visible(fomdgt3_7,359). happensAt(walking(fomdgt3_7),359).
orientation(fomdgt3_2,42,360). coords(fomdgt3_2,262,93,360).  visible(fomdgt3_2,360). happensAt(walking(fomdgt3_2),360). orientation(fomdgt3_7,134,360). coords(fomdgt3_7,78,76,360).  visible(fomdgt3_7,360). happensAt(walking(fomdgt3_7),360).
orientation(fomdgt3_2,42,361). coords(fomdgt3_2,262,93,361).  visible(fomdgt3_2,361). happensAt(walking(fomdgt3_2),361). orientation(fomdgt3_7,134,361). coords(fomdgt3_7,78,76,361).  visible(fomdgt3_7,361). happensAt(walking(fomdgt3_7),361).
orientation(fomdgt3_2,42,362). coords(fomdgt3_2,262,95,362).  visible(fomdgt3_2,362). happensAt(walking(fomdgt3_2),362). orientation(fomdgt3_7,134,362). coords(fomdgt3_7,78,76,362).  visible(fomdgt3_7,362). happensAt(walking(fomdgt3_7),362).
#show holdsAt/2.