
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

	orientation(mws1gt_1,143,352). coords(mws1gt_1,70,92,352).  visible(mws1gt_1,352). happensAt(walking(mws1gt_1),352). orientation(mws1gt_6,160,352). coords(mws1gt_6,32,198,352).  visible(mws1gt_6,352). happensAt(walking(mws1gt_6),352).
orientation(mws1gt_1,143,353). coords(mws1gt_1,70,92,353).  visible(mws1gt_1,353). happensAt(walking(mws1gt_1),353). orientation(mws1gt_6,160,353). coords(mws1gt_6,32,199,353).  visible(mws1gt_6,353). happensAt(walking(mws1gt_6),353).
orientation(mws1gt_1,143,354). coords(mws1gt_1,70,92,354).  visible(mws1gt_1,354). happensAt(walking(mws1gt_1),354). orientation(mws1gt_6,160,354). coords(mws1gt_6,32,200,354).  visible(mws1gt_6,354). happensAt(walking(mws1gt_6),354).
orientation(mws1gt_1,143,355). coords(mws1gt_1,70,91,355).  visible(mws1gt_1,355). happensAt(walking(mws1gt_1),355). orientation(mws1gt_6,160,355). coords(mws1gt_6,32,201,355).  visible(mws1gt_6,355). happensAt(walking(mws1gt_6),355).
orientation(mws1gt_1,143,356). coords(mws1gt_1,69,90,356).  visible(mws1gt_1,356). happensAt(walking(mws1gt_1),356). orientation(mws1gt_6,160,356). coords(mws1gt_6,32,201,356).  visible(mws1gt_6,356). happensAt(walking(mws1gt_6),356).
orientation(mws1gt_1,143,357). coords(mws1gt_1,69,89,357).  visible(mws1gt_1,357). happensAt(walking(mws1gt_1),357). orientation(mws1gt_6,160,357). coords(mws1gt_6,32,201,357).  visible(mws1gt_6,357). happensAt(walking(mws1gt_6),357).
orientation(mws1gt_1,143,358). coords(mws1gt_1,69,88,358).  visible(mws1gt_1,358). happensAt(walking(mws1gt_1),358). orientation(mws1gt_6,160,358). coords(mws1gt_6,32,201,358).  visible(mws1gt_6,358). happensAt(walking(mws1gt_6),358).
orientation(mws1gt_1,143,359). coords(mws1gt_1,69,88,359).  visible(mws1gt_1,359). happensAt(walking(mws1gt_1),359). orientation(mws1gt_6,160,359). coords(mws1gt_6,32,202,359).  visible(mws1gt_6,359). happensAt(walking(mws1gt_6),359).
orientation(mws1gt_1,143,360). coords(mws1gt_1,69,87,360).  visible(mws1gt_1,360). happensAt(walking(mws1gt_1),360). orientation(mws1gt_6,160,360). coords(mws1gt_6,32,203,360).  visible(mws1gt_6,360). happensAt(walking(mws1gt_6),360).
orientation(mws1gt_1,143,361). coords(mws1gt_1,69,87,361).  visible(mws1gt_1,361). happensAt(walking(mws1gt_1),361). orientation(mws1gt_6,160,361). coords(mws1gt_6,32,203,361).  visible(mws1gt_6,361). happensAt(walking(mws1gt_6),361).
orientation(mws1gt_1,143,362). coords(mws1gt_1,69,87,362).  visible(mws1gt_1,362). happensAt(walking(mws1gt_1),362). orientation(mws1gt_6,160,362). coords(mws1gt_6,32,204,362).  visible(mws1gt_6,362). happensAt(walking(mws1gt_6),362).
orientation(mws1gt_1,143,363). coords(mws1gt_1,69,86,363).  visible(mws1gt_1,363). happensAt(walking(mws1gt_1),363). orientation(mws1gt_6,160,363). coords(mws1gt_6,31,204,363).  visible(mws1gt_6,363). happensAt(walking(mws1gt_6),363).
orientation(mws1gt_1,143,364). coords(mws1gt_1,69,86,364).  visible(mws1gt_1,364). happensAt(walking(mws1gt_1),364). orientation(mws1gt_6,160,364). coords(mws1gt_6,31,205,364).  visible(mws1gt_6,364). happensAt(walking(mws1gt_6),364).
orientation(mws1gt_1,143,365). coords(mws1gt_1,70,85,365).  visible(mws1gt_1,365). happensAt(walking(mws1gt_1),365). orientation(mws1gt_6,160,365). coords(mws1gt_6,30,205,365).  visible(mws1gt_6,365). happensAt(walking(mws1gt_6),365).
orientation(mws1gt_1,143,366). coords(mws1gt_1,70,85,366).  visible(mws1gt_1,366). happensAt(walking(mws1gt_1),366). orientation(mws1gt_6,160,366). coords(mws1gt_6,30,206,366).  visible(mws1gt_6,366). happensAt(walking(mws1gt_6),366).
orientation(mws1gt_1,143,367). coords(mws1gt_1,70,84,367).  visible(mws1gt_1,367). happensAt(walking(mws1gt_1),367). orientation(mws1gt_6,160,367). coords(mws1gt_6,30,206,367).  visible(mws1gt_6,367). happensAt(walking(mws1gt_6),367).
orientation(mws1gt_1,143,368). coords(mws1gt_1,70,84,368).  visible(mws1gt_1,368). happensAt(walking(mws1gt_1),368). orientation(mws1gt_6,160,368). coords(mws1gt_6,30,207,368).  visible(mws1gt_6,368). happensAt(walking(mws1gt_6),368).
orientation(mws1gt_1,143,369). coords(mws1gt_1,70,84,369).  visible(mws1gt_1,369). happensAt(walking(mws1gt_1),369). orientation(mws1gt_6,160,369). coords(mws1gt_6,29,208,369).  visible(mws1gt_6,369). happensAt(walking(mws1gt_6),369).
orientation(mws1gt_1,143,370). coords(mws1gt_1,70,84,370).  visible(mws1gt_1,370). happensAt(walking(mws1gt_1),370). orientation(mws1gt_6,160,370). coords(mws1gt_6,27,209,370).  visible(mws1gt_6,370). happensAt(walking(mws1gt_6),370).
orientation(mws1gt_1,143,371). coords(mws1gt_1,70,82,371).  visible(mws1gt_1,371). happensAt(walking(mws1gt_1),371). orientation(mws1gt_6,160,371). coords(mws1gt_6,27,210,371).  visible(mws1gt_6,371). happensAt(walking(mws1gt_6),371).
orientation(mws1gt_1,143,372). coords(mws1gt_1,70,81,372).  visible(mws1gt_1,372). happensAt(walking(mws1gt_1),372). orientation(mws1gt_6,160,372). coords(mws1gt_6,26,210,372).  visible(mws1gt_6,372). happensAt(walking(mws1gt_6),372).
orientation(mws1gt_1,143,373). coords(mws1gt_1,70,80,373).  visible(mws1gt_1,373). happensAt(walking(mws1gt_1),373). orientation(mws1gt_6,160,373). coords(mws1gt_6,26,211,373).  visible(mws1gt_6,373). happensAt(walking(mws1gt_6),373).
orientation(mws1gt_1,143,374). coords(mws1gt_1,70,79,374).  visible(mws1gt_1,374). happensAt(walking(mws1gt_1),374). orientation(mws1gt_6,160,374). coords(mws1gt_6,25,211,374).  visible(mws1gt_6,374). happensAt(walking(mws1gt_6),374).
orientation(mws1gt_1,143,375). coords(mws1gt_1,70,78,375).  visible(mws1gt_1,375). happensAt(walking(mws1gt_1),375). orientation(mws1gt_6,160,375). coords(mws1gt_6,24,211,375).  visible(mws1gt_6,375). happensAt(walking(mws1gt_6),375).
orientation(mws1gt_1,143,376). coords(mws1gt_1,70,77,376).  visible(mws1gt_1,376). happensAt(walking(mws1gt_1),376). orientation(mws1gt_6,160,376). coords(mws1gt_6,24,212,376).  happensAt(disappear(mws1gt_6),376). happensAt(walking(mws1gt_6),376).
#show holdsAt/2.