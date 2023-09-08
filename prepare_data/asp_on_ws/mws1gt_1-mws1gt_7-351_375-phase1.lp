
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

	orientation(mws1gt_1,143,351). coords(mws1gt_1,70,92,351).  visible(mws1gt_1,351). happensAt(walking(mws1gt_1),351). orientation(mws1gt_7,174,351). coords(mws1gt_7,46,188,351).  visible(mws1gt_7,351). happensAt(walking(mws1gt_7),351).
orientation(mws1gt_1,143,352). coords(mws1gt_1,70,92,352).  visible(mws1gt_1,352). happensAt(walking(mws1gt_1),352). orientation(mws1gt_7,174,352). coords(mws1gt_7,46,188,352).  visible(mws1gt_7,352). happensAt(walking(mws1gt_7),352).
orientation(mws1gt_1,143,353). coords(mws1gt_1,70,92,353).  visible(mws1gt_1,353). happensAt(walking(mws1gt_1),353). orientation(mws1gt_7,174,353). coords(mws1gt_7,46,188,353).  visible(mws1gt_7,353). happensAt(walking(mws1gt_7),353).
orientation(mws1gt_1,143,354). coords(mws1gt_1,70,92,354).  visible(mws1gt_1,354). happensAt(walking(mws1gt_1),354). orientation(mws1gt_7,174,354). coords(mws1gt_7,46,188,354).  visible(mws1gt_7,354). happensAt(walking(mws1gt_7),354).
orientation(mws1gt_1,143,355). coords(mws1gt_1,70,91,355).  visible(mws1gt_1,355). happensAt(walking(mws1gt_1),355). orientation(mws1gt_7,174,355). coords(mws1gt_7,46,187,355).  visible(mws1gt_7,355). happensAt(walking(mws1gt_7),355).
orientation(mws1gt_1,143,356). coords(mws1gt_1,69,90,356).  visible(mws1gt_1,356). happensAt(walking(mws1gt_1),356). orientation(mws1gt_7,174,356). coords(mws1gt_7,46,187,356).  visible(mws1gt_7,356). happensAt(walking(mws1gt_7),356).
orientation(mws1gt_1,143,357). coords(mws1gt_1,69,89,357).  visible(mws1gt_1,357). happensAt(walking(mws1gt_1),357). orientation(mws1gt_7,174,357). coords(mws1gt_7,47,187,357).  visible(mws1gt_7,357). happensAt(walking(mws1gt_7),357).
orientation(mws1gt_1,143,358). coords(mws1gt_1,69,88,358).  visible(mws1gt_1,358). happensAt(walking(mws1gt_1),358). orientation(mws1gt_7,174,358). coords(mws1gt_7,47,187,358).  visible(mws1gt_7,358). happensAt(walking(mws1gt_7),358).
orientation(mws1gt_1,143,359). coords(mws1gt_1,69,88,359).  visible(mws1gt_1,359). happensAt(walking(mws1gt_1),359). orientation(mws1gt_7,174,359). coords(mws1gt_7,49,186,359).  visible(mws1gt_7,359). happensAt(walking(mws1gt_7),359).
orientation(mws1gt_1,143,360). coords(mws1gt_1,69,87,360).  visible(mws1gt_1,360). happensAt(walking(mws1gt_1),360). orientation(mws1gt_7,174,360). coords(mws1gt_7,50,185,360).  visible(mws1gt_7,360). happensAt(walking(mws1gt_7),360).
orientation(mws1gt_1,143,361). coords(mws1gt_1,69,87,361).  visible(mws1gt_1,361). happensAt(walking(mws1gt_1),361). orientation(mws1gt_7,174,361). coords(mws1gt_7,50,185,361).  visible(mws1gt_7,361). happensAt(walking(mws1gt_7),361).
orientation(mws1gt_1,143,362). coords(mws1gt_1,69,87,362).  visible(mws1gt_1,362). happensAt(walking(mws1gt_1),362). orientation(mws1gt_7,174,362). coords(mws1gt_7,50,185,362).  visible(mws1gt_7,362). happensAt(walking(mws1gt_7),362).
orientation(mws1gt_1,143,363). coords(mws1gt_1,69,86,363).  visible(mws1gt_1,363). happensAt(walking(mws1gt_1),363). orientation(mws1gt_7,174,363). coords(mws1gt_7,50,185,363).  visible(mws1gt_7,363). happensAt(walking(mws1gt_7),363).
orientation(mws1gt_1,143,364). coords(mws1gt_1,69,86,364).  visible(mws1gt_1,364). happensAt(walking(mws1gt_1),364). orientation(mws1gt_7,174,364). coords(mws1gt_7,50,185,364).  visible(mws1gt_7,364). happensAt(walking(mws1gt_7),364).
orientation(mws1gt_1,143,365). coords(mws1gt_1,70,85,365).  visible(mws1gt_1,365). happensAt(walking(mws1gt_1),365). orientation(mws1gt_7,174,365). coords(mws1gt_7,50,185,365).  visible(mws1gt_7,365). happensAt(walking(mws1gt_7),365).
orientation(mws1gt_1,143,366). coords(mws1gt_1,70,85,366).  visible(mws1gt_1,366). happensAt(walking(mws1gt_1),366). orientation(mws1gt_7,174,366). coords(mws1gt_7,50,185,366).  visible(mws1gt_7,366). happensAt(walking(mws1gt_7),366).
orientation(mws1gt_1,143,367). coords(mws1gt_1,70,84,367).  visible(mws1gt_1,367). happensAt(walking(mws1gt_1),367). orientation(mws1gt_7,174,367). coords(mws1gt_7,50,185,367).  visible(mws1gt_7,367). happensAt(walking(mws1gt_7),367).
orientation(mws1gt_1,143,368). coords(mws1gt_1,70,84,368).  visible(mws1gt_1,368). happensAt(walking(mws1gt_1),368). orientation(mws1gt_7,174,368). coords(mws1gt_7,50,185,368).  visible(mws1gt_7,368). happensAt(walking(mws1gt_7),368).
orientation(mws1gt_1,143,369). coords(mws1gt_1,70,84,369).  visible(mws1gt_1,369). happensAt(walking(mws1gt_1),369). orientation(mws1gt_7,174,369). coords(mws1gt_7,50,185,369).  visible(mws1gt_7,369). happensAt(walking(mws1gt_7),369).
orientation(mws1gt_1,143,370). coords(mws1gt_1,70,84,370).  visible(mws1gt_1,370). happensAt(walking(mws1gt_1),370). orientation(mws1gt_7,174,370). coords(mws1gt_7,50,185,370).  visible(mws1gt_7,370). happensAt(walking(mws1gt_7),370).
orientation(mws1gt_1,143,371). coords(mws1gt_1,70,82,371).  visible(mws1gt_1,371). happensAt(walking(mws1gt_1),371). orientation(mws1gt_7,174,371). coords(mws1gt_7,51,185,371).  visible(mws1gt_7,371). happensAt(walking(mws1gt_7),371).
orientation(mws1gt_1,143,372). coords(mws1gt_1,70,81,372).  visible(mws1gt_1,372). happensAt(walking(mws1gt_1),372). orientation(mws1gt_7,174,372). coords(mws1gt_7,53,185,372).  visible(mws1gt_7,372). happensAt(walking(mws1gt_7),372).
orientation(mws1gt_1,143,373). coords(mws1gt_1,70,80,373).  visible(mws1gt_1,373). happensAt(walking(mws1gt_1),373). orientation(mws1gt_7,174,373). coords(mws1gt_7,53,185,373).  visible(mws1gt_7,373). happensAt(walking(mws1gt_7),373).
orientation(mws1gt_1,143,374). coords(mws1gt_1,70,79,374).  visible(mws1gt_1,374). happensAt(walking(mws1gt_1),374). orientation(mws1gt_7,174,374). coords(mws1gt_7,55,185,374).  visible(mws1gt_7,374). happensAt(walking(mws1gt_7),374).
orientation(mws1gt_1,143,375). coords(mws1gt_1,70,78,375).  visible(mws1gt_1,375). happensAt(walking(mws1gt_1),375). orientation(mws1gt_7,174,375). coords(mws1gt_7,56,184,375).  visible(mws1gt_7,375). happensAt(walking(mws1gt_7),375).
#show holdsAt/2.