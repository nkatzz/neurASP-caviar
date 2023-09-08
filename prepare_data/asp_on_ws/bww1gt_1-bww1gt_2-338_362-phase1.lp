
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

	orientation(bww1gt_1,7,338). coords(bww1gt_1,105,264,338).  visible(bww1gt_1,338). happensAt(inactive(bww1gt_1),338). orientation(bww1gt_2,167,338). coords(bww1gt_2,75,159,338).  visible(bww1gt_2,338). happensAt(walking(bww1gt_2),338).
orientation(bww1gt_1,9,339). coords(bww1gt_1,105,264,339).  visible(bww1gt_1,339). happensAt(inactive(bww1gt_1),339). orientation(bww1gt_2,167,339). coords(bww1gt_2,75,159,339).  visible(bww1gt_2,339). happensAt(walking(bww1gt_2),339).
orientation(bww1gt_1,6,340). coords(bww1gt_1,105,264,340).  visible(bww1gt_1,340). happensAt(inactive(bww1gt_1),340). orientation(bww1gt_2,167,340). coords(bww1gt_2,76,159,340).  visible(bww1gt_2,340). happensAt(walking(bww1gt_2),340).
orientation(bww1gt_1,5,341). coords(bww1gt_1,105,264,341).  visible(bww1gt_1,341). happensAt(inactive(bww1gt_1),341). orientation(bww1gt_2,167,341). coords(bww1gt_2,76,159,341).  visible(bww1gt_2,341). happensAt(walking(bww1gt_2),341).
orientation(bww1gt_1,8,342). coords(bww1gt_1,105,264,342).  visible(bww1gt_1,342). happensAt(inactive(bww1gt_1),342). orientation(bww1gt_2,167,342). coords(bww1gt_2,76,159,342).  visible(bww1gt_2,342). happensAt(walking(bww1gt_2),342).
orientation(bww1gt_1,8,343). coords(bww1gt_1,105,264,343).  visible(bww1gt_1,343). happensAt(inactive(bww1gt_1),343). orientation(bww1gt_2,167,343). coords(bww1gt_2,76,159,343).  visible(bww1gt_2,343). happensAt(walking(bww1gt_2),343).
orientation(bww1gt_1,10,344). coords(bww1gt_1,105,264,344).  visible(bww1gt_1,344). happensAt(inactive(bww1gt_1),344). orientation(bww1gt_2,167,344). coords(bww1gt_2,76,159,344).  visible(bww1gt_2,344). happensAt(walking(bww1gt_2),344).
orientation(bww1gt_1,7,345). coords(bww1gt_1,105,264,345).  visible(bww1gt_1,345). happensAt(inactive(bww1gt_1),345). orientation(bww1gt_2,167,345). coords(bww1gt_2,76,159,345).  visible(bww1gt_2,345). happensAt(walking(bww1gt_2),345).
orientation(bww1gt_1,8,346). coords(bww1gt_1,105,264,346).  visible(bww1gt_1,346). happensAt(inactive(bww1gt_1),346). orientation(bww1gt_2,167,346). coords(bww1gt_2,76,159,346).  visible(bww1gt_2,346). happensAt(walking(bww1gt_2),346).
orientation(bww1gt_1,4,347). coords(bww1gt_1,105,264,347).  visible(bww1gt_1,347). happensAt(inactive(bww1gt_1),347). orientation(bww1gt_2,167,347). coords(bww1gt_2,77,159,347).  visible(bww1gt_2,347). happensAt(walking(bww1gt_2),347).
orientation(bww1gt_1,0,348). coords(bww1gt_1,105,264,348).  visible(bww1gt_1,348). happensAt(inactive(bww1gt_1),348). orientation(bww1gt_2,167,348). coords(bww1gt_2,77,158,348).  visible(bww1gt_2,348). happensAt(walking(bww1gt_2),348).
orientation(bww1gt_1,5,349). coords(bww1gt_1,105,264,349).  visible(bww1gt_1,349). happensAt(inactive(bww1gt_1),349). orientation(bww1gt_2,167,349). coords(bww1gt_2,77,158,349).  visible(bww1gt_2,349). happensAt(walking(bww1gt_2),349).
orientation(bww1gt_1,3,350). coords(bww1gt_1,105,264,350).  visible(bww1gt_1,350). happensAt(inactive(bww1gt_1),350). orientation(bww1gt_2,167,350). coords(bww1gt_2,78,158,350).  visible(bww1gt_2,350). happensAt(walking(bww1gt_2),350).
orientation(bww1gt_1,4,351). coords(bww1gt_1,105,264,351).  visible(bww1gt_1,351). happensAt(inactive(bww1gt_1),351). orientation(bww1gt_2,167,351). coords(bww1gt_2,80,158,351).  visible(bww1gt_2,351). happensAt(walking(bww1gt_2),351).
orientation(bww1gt_1,4,352). coords(bww1gt_1,105,264,352).  visible(bww1gt_1,352). happensAt(inactive(bww1gt_1),352). orientation(bww1gt_2,167,352). coords(bww1gt_2,82,158,352).  visible(bww1gt_2,352). happensAt(walking(bww1gt_2),352).
orientation(bww1gt_1,4,353). coords(bww1gt_1,105,264,353).  visible(bww1gt_1,353). happensAt(inactive(bww1gt_1),353). orientation(bww1gt_2,167,353). coords(bww1gt_2,83,158,353).  visible(bww1gt_2,353). happensAt(walking(bww1gt_2),353).
orientation(bww1gt_1,7,354). coords(bww1gt_1,105,264,354).  visible(bww1gt_1,354). happensAt(inactive(bww1gt_1),354). orientation(bww1gt_2,167,354). coords(bww1gt_2,84,158,354).  visible(bww1gt_2,354). happensAt(walking(bww1gt_2),354).
orientation(bww1gt_1,4,355). coords(bww1gt_1,105,264,355).  visible(bww1gt_1,355). happensAt(inactive(bww1gt_1),355). orientation(bww1gt_2,167,355). coords(bww1gt_2,84,157,355).  visible(bww1gt_2,355). happensAt(walking(bww1gt_2),355).
orientation(bww1gt_1,2,356). coords(bww1gt_1,105,264,356).  visible(bww1gt_1,356). happensAt(inactive(bww1gt_1),356). orientation(bww1gt_2,167,356). coords(bww1gt_2,85,157,356).  visible(bww1gt_2,356). happensAt(walking(bww1gt_2),356).
orientation(bww1gt_1,3,357). coords(bww1gt_1,105,264,357).  visible(bww1gt_1,357). happensAt(inactive(bww1gt_1),357). orientation(bww1gt_2,167,357). coords(bww1gt_2,85,157,357).  visible(bww1gt_2,357). happensAt(walking(bww1gt_2),357).
orientation(bww1gt_1,5,358). coords(bww1gt_1,105,264,358).  visible(bww1gt_1,358). happensAt(inactive(bww1gt_1),358). orientation(bww1gt_2,167,358). coords(bww1gt_2,85,156,358).  visible(bww1gt_2,358). happensAt(walking(bww1gt_2),358).
orientation(bww1gt_1,2,359). coords(bww1gt_1,105,264,359).  visible(bww1gt_1,359). happensAt(inactive(bww1gt_1),359). orientation(bww1gt_2,167,359). coords(bww1gt_2,85,156,359).  visible(bww1gt_2,359). happensAt(walking(bww1gt_2),359).
orientation(bww1gt_1,3,360). coords(bww1gt_1,105,264,360).  visible(bww1gt_1,360). happensAt(inactive(bww1gt_1),360). orientation(bww1gt_2,167,360). coords(bww1gt_2,86,155,360).  visible(bww1gt_2,360). happensAt(walking(bww1gt_2),360).
orientation(bww1gt_1,5,361). coords(bww1gt_1,105,264,361).  visible(bww1gt_1,361). happensAt(inactive(bww1gt_1),361). orientation(bww1gt_2,157,361). coords(bww1gt_2,86,155,361).  visible(bww1gt_2,361). happensAt(walking(bww1gt_2),361).
orientation(bww1gt_1,4,362). coords(bww1gt_1,105,264,362).  visible(bww1gt_1,362). happensAt(inactive(bww1gt_1),362). orientation(bww1gt_2,156,362). coords(bww1gt_2,87,155,362).  visible(bww1gt_2,362). happensAt(walking(bww1gt_2),362).
#show holdsAt/2.