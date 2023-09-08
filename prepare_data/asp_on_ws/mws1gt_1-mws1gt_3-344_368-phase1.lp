
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

	orientation(mws1gt_1,143,344). coords(mws1gt_1,71,95,344).  visible(mws1gt_1,344). happensAt(walking(mws1gt_1),344). orientation(mws1gt_3,165,344). coords(mws1gt_3,55,151,344).  visible(mws1gt_3,344). happensAt(walking(mws1gt_3),344).
orientation(mws1gt_1,143,345). coords(mws1gt_1,70,94,345).  visible(mws1gt_1,345). happensAt(walking(mws1gt_1),345). orientation(mws1gt_3,165,345). coords(mws1gt_3,55,151,345).  visible(mws1gt_3,345). happensAt(walking(mws1gt_3),345).
orientation(mws1gt_1,143,346). coords(mws1gt_1,70,94,346).  visible(mws1gt_1,346). happensAt(walking(mws1gt_1),346). orientation(mws1gt_3,165,346). coords(mws1gt_3,54,152,346).  visible(mws1gt_3,346). happensAt(walking(mws1gt_3),346).
orientation(mws1gt_1,143,347). coords(mws1gt_1,70,93,347).  visible(mws1gt_1,347). happensAt(walking(mws1gt_1),347). orientation(mws1gt_3,165,347). coords(mws1gt_3,54,152,347).  visible(mws1gt_3,347). happensAt(walking(mws1gt_3),347).
orientation(mws1gt_1,143,348). coords(mws1gt_1,70,93,348).  visible(mws1gt_1,348). happensAt(walking(mws1gt_1),348). orientation(mws1gt_3,165,348). coords(mws1gt_3,54,152,348).  visible(mws1gt_3,348). happensAt(walking(mws1gt_3),348).
orientation(mws1gt_1,143,349). coords(mws1gt_1,70,93,349).  visible(mws1gt_1,349). happensAt(walking(mws1gt_1),349). orientation(mws1gt_3,165,349). coords(mws1gt_3,54,153,349).  visible(mws1gt_3,349). happensAt(walking(mws1gt_3),349).
orientation(mws1gt_1,143,350). coords(mws1gt_1,70,92,350).  visible(mws1gt_1,350). happensAt(walking(mws1gt_1),350). orientation(mws1gt_3,165,350). coords(mws1gt_3,54,154,350).  visible(mws1gt_3,350). happensAt(walking(mws1gt_3),350).
orientation(mws1gt_1,143,351). coords(mws1gt_1,70,92,351).  visible(mws1gt_1,351). happensAt(walking(mws1gt_1),351). orientation(mws1gt_3,165,351). coords(mws1gt_3,53,154,351).  visible(mws1gt_3,351). happensAt(walking(mws1gt_3),351).
orientation(mws1gt_1,143,352). coords(mws1gt_1,70,92,352).  visible(mws1gt_1,352). happensAt(walking(mws1gt_1),352). orientation(mws1gt_3,165,352). coords(mws1gt_3,53,154,352).  visible(mws1gt_3,352). happensAt(walking(mws1gt_3),352).
orientation(mws1gt_1,143,353). coords(mws1gt_1,70,92,353).  visible(mws1gt_1,353). happensAt(walking(mws1gt_1),353). orientation(mws1gt_3,165,353). coords(mws1gt_3,53,154,353).  visible(mws1gt_3,353). happensAt(walking(mws1gt_3),353).
orientation(mws1gt_1,143,354). coords(mws1gt_1,70,92,354).  visible(mws1gt_1,354). happensAt(walking(mws1gt_1),354). orientation(mws1gt_3,165,354). coords(mws1gt_3,52,154,354).  visible(mws1gt_3,354). happensAt(walking(mws1gt_3),354).
orientation(mws1gt_1,143,355). coords(mws1gt_1,70,91,355).  visible(mws1gt_1,355). happensAt(walking(mws1gt_1),355). orientation(mws1gt_3,165,355). coords(mws1gt_3,51,154,355).  visible(mws1gt_3,355). happensAt(walking(mws1gt_3),355).
orientation(mws1gt_1,143,356). coords(mws1gt_1,69,90,356).  visible(mws1gt_1,356). happensAt(walking(mws1gt_1),356). orientation(mws1gt_3,165,356). coords(mws1gt_3,51,155,356).  visible(mws1gt_3,356). happensAt(walking(mws1gt_3),356).
orientation(mws1gt_1,143,357). coords(mws1gt_1,69,89,357).  visible(mws1gt_1,357). happensAt(walking(mws1gt_1),357). orientation(mws1gt_3,165,357). coords(mws1gt_3,50,155,357).  visible(mws1gt_3,357). happensAt(walking(mws1gt_3),357).
orientation(mws1gt_1,143,358). coords(mws1gt_1,69,88,358).  visible(mws1gt_1,358). happensAt(walking(mws1gt_1),358). orientation(mws1gt_3,165,358). coords(mws1gt_3,50,157,358).  visible(mws1gt_3,358). happensAt(walking(mws1gt_3),358).
orientation(mws1gt_1,143,359). coords(mws1gt_1,69,88,359).  visible(mws1gt_1,359). happensAt(walking(mws1gt_1),359). orientation(mws1gt_3,165,359). coords(mws1gt_3,49,157,359).  visible(mws1gt_3,359). happensAt(walking(mws1gt_3),359).
orientation(mws1gt_1,143,360). coords(mws1gt_1,69,87,360).  visible(mws1gt_1,360). happensAt(walking(mws1gt_1),360). orientation(mws1gt_3,165,360). coords(mws1gt_3,49,158,360).  visible(mws1gt_3,360). happensAt(walking(mws1gt_3),360).
orientation(mws1gt_1,143,361). coords(mws1gt_1,69,87,361).  visible(mws1gt_1,361). happensAt(walking(mws1gt_1),361). orientation(mws1gt_3,165,361). coords(mws1gt_3,49,159,361).  visible(mws1gt_3,361). happensAt(walking(mws1gt_3),361).
orientation(mws1gt_1,143,362). coords(mws1gt_1,69,87,362).  visible(mws1gt_1,362). happensAt(walking(mws1gt_1),362). orientation(mws1gt_3,165,362). coords(mws1gt_3,49,159,362).  visible(mws1gt_3,362). happensAt(walking(mws1gt_3),362).
orientation(mws1gt_1,143,363). coords(mws1gt_1,69,86,363).  visible(mws1gt_1,363). happensAt(walking(mws1gt_1),363). orientation(mws1gt_3,165,363). coords(mws1gt_3,49,159,363).  visible(mws1gt_3,363). happensAt(walking(mws1gt_3),363).
orientation(mws1gt_1,143,364). coords(mws1gt_1,69,86,364).  visible(mws1gt_1,364). happensAt(walking(mws1gt_1),364). orientation(mws1gt_3,165,364). coords(mws1gt_3,48,159,364).  visible(mws1gt_3,364). happensAt(walking(mws1gt_3),364).
orientation(mws1gt_1,143,365). coords(mws1gt_1,70,85,365).  visible(mws1gt_1,365). happensAt(walking(mws1gt_1),365). orientation(mws1gt_3,165,365). coords(mws1gt_3,47,159,365).  visible(mws1gt_3,365). happensAt(walking(mws1gt_3),365).
orientation(mws1gt_1,143,366). coords(mws1gt_1,70,85,366).  visible(mws1gt_1,366). happensAt(walking(mws1gt_1),366). orientation(mws1gt_3,165,366). coords(mws1gt_3,47,159,366).  visible(mws1gt_3,366). happensAt(walking(mws1gt_3),366).
orientation(mws1gt_1,143,367). coords(mws1gt_1,70,84,367).  visible(mws1gt_1,367). happensAt(walking(mws1gt_1),367). orientation(mws1gt_3,165,367). coords(mws1gt_3,47,159,367).  visible(mws1gt_3,367). happensAt(walking(mws1gt_3),367).
orientation(mws1gt_1,143,368). coords(mws1gt_1,70,84,368).  visible(mws1gt_1,368). happensAt(walking(mws1gt_1),368). orientation(mws1gt_3,165,368). coords(mws1gt_3,46,159,368).  visible(mws1gt_3,368). happensAt(walking(mws1gt_3),368).
#show holdsAt/2.