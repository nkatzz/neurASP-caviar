
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

	orientation(mws1gt_3,165,352). coords(mws1gt_3,53,154,352).  visible(mws1gt_3,352). happensAt(walking(mws1gt_3),352). orientation(mws1gt_6,160,352). coords(mws1gt_6,32,198,352).  visible(mws1gt_6,352). happensAt(walking(mws1gt_6),352).
orientation(mws1gt_3,165,353). coords(mws1gt_3,53,154,353).  visible(mws1gt_3,353). happensAt(walking(mws1gt_3),353). orientation(mws1gt_6,160,353). coords(mws1gt_6,32,199,353).  visible(mws1gt_6,353). happensAt(walking(mws1gt_6),353).
orientation(mws1gt_3,165,354). coords(mws1gt_3,52,154,354).  visible(mws1gt_3,354). happensAt(walking(mws1gt_3),354). orientation(mws1gt_6,160,354). coords(mws1gt_6,32,200,354).  visible(mws1gt_6,354). happensAt(walking(mws1gt_6),354).
orientation(mws1gt_3,165,355). coords(mws1gt_3,51,154,355).  visible(mws1gt_3,355). happensAt(walking(mws1gt_3),355). orientation(mws1gt_6,160,355). coords(mws1gt_6,32,201,355).  visible(mws1gt_6,355). happensAt(walking(mws1gt_6),355).
orientation(mws1gt_3,165,356). coords(mws1gt_3,51,155,356).  visible(mws1gt_3,356). happensAt(walking(mws1gt_3),356). orientation(mws1gt_6,160,356). coords(mws1gt_6,32,201,356).  visible(mws1gt_6,356). happensAt(walking(mws1gt_6),356).
orientation(mws1gt_3,165,357). coords(mws1gt_3,50,155,357).  visible(mws1gt_3,357). happensAt(walking(mws1gt_3),357). orientation(mws1gt_6,160,357). coords(mws1gt_6,32,201,357).  visible(mws1gt_6,357). happensAt(walking(mws1gt_6),357).
orientation(mws1gt_3,165,358). coords(mws1gt_3,50,157,358).  visible(mws1gt_3,358). happensAt(walking(mws1gt_3),358). orientation(mws1gt_6,160,358). coords(mws1gt_6,32,201,358).  visible(mws1gt_6,358). happensAt(walking(mws1gt_6),358).
orientation(mws1gt_3,165,359). coords(mws1gt_3,49,157,359).  visible(mws1gt_3,359). happensAt(walking(mws1gt_3),359). orientation(mws1gt_6,160,359). coords(mws1gt_6,32,202,359).  visible(mws1gt_6,359). happensAt(walking(mws1gt_6),359).
orientation(mws1gt_3,165,360). coords(mws1gt_3,49,158,360).  visible(mws1gt_3,360). happensAt(walking(mws1gt_3),360). orientation(mws1gt_6,160,360). coords(mws1gt_6,32,203,360).  visible(mws1gt_6,360). happensAt(walking(mws1gt_6),360).
orientation(mws1gt_3,165,361). coords(mws1gt_3,49,159,361).  visible(mws1gt_3,361). happensAt(walking(mws1gt_3),361). orientation(mws1gt_6,160,361). coords(mws1gt_6,32,203,361).  visible(mws1gt_6,361). happensAt(walking(mws1gt_6),361).
orientation(mws1gt_3,165,362). coords(mws1gt_3,49,159,362).  visible(mws1gt_3,362). happensAt(walking(mws1gt_3),362). orientation(mws1gt_6,160,362). coords(mws1gt_6,32,204,362).  visible(mws1gt_6,362). happensAt(walking(mws1gt_6),362).
orientation(mws1gt_3,165,363). coords(mws1gt_3,49,159,363).  visible(mws1gt_3,363). happensAt(walking(mws1gt_3),363). orientation(mws1gt_6,160,363). coords(mws1gt_6,31,204,363).  visible(mws1gt_6,363). happensAt(walking(mws1gt_6),363).
orientation(mws1gt_3,165,364). coords(mws1gt_3,48,159,364).  visible(mws1gt_3,364). happensAt(walking(mws1gt_3),364). orientation(mws1gt_6,160,364). coords(mws1gt_6,31,205,364).  visible(mws1gt_6,364). happensAt(walking(mws1gt_6),364).
orientation(mws1gt_3,165,365). coords(mws1gt_3,47,159,365).  visible(mws1gt_3,365). happensAt(walking(mws1gt_3),365). orientation(mws1gt_6,160,365). coords(mws1gt_6,30,205,365).  visible(mws1gt_6,365). happensAt(walking(mws1gt_6),365).
orientation(mws1gt_3,165,366). coords(mws1gt_3,47,159,366).  visible(mws1gt_3,366). happensAt(walking(mws1gt_3),366). orientation(mws1gt_6,160,366). coords(mws1gt_6,30,206,366).  visible(mws1gt_6,366). happensAt(walking(mws1gt_6),366).
orientation(mws1gt_3,165,367). coords(mws1gt_3,47,159,367).  visible(mws1gt_3,367). happensAt(walking(mws1gt_3),367). orientation(mws1gt_6,160,367). coords(mws1gt_6,30,206,367).  visible(mws1gt_6,367). happensAt(walking(mws1gt_6),367).
orientation(mws1gt_3,165,368). coords(mws1gt_3,46,159,368).  visible(mws1gt_3,368). happensAt(walking(mws1gt_3),368). orientation(mws1gt_6,160,368). coords(mws1gt_6,30,207,368).  visible(mws1gt_6,368). happensAt(walking(mws1gt_6),368).
orientation(mws1gt_3,165,369). coords(mws1gt_3,45,160,369).  visible(mws1gt_3,369). happensAt(walking(mws1gt_3),369). orientation(mws1gt_6,160,369). coords(mws1gt_6,29,208,369).  visible(mws1gt_6,369). happensAt(walking(mws1gt_6),369).
orientation(mws1gt_3,165,370). coords(mws1gt_3,45,160,370).  visible(mws1gt_3,370). happensAt(walking(mws1gt_3),370). orientation(mws1gt_6,160,370). coords(mws1gt_6,27,209,370).  visible(mws1gt_6,370). happensAt(walking(mws1gt_6),370).
orientation(mws1gt_3,165,371). coords(mws1gt_3,44,160,371).  visible(mws1gt_3,371). happensAt(walking(mws1gt_3),371). orientation(mws1gt_6,160,371). coords(mws1gt_6,27,210,371).  visible(mws1gt_6,371). happensAt(walking(mws1gt_6),371).
orientation(mws1gt_3,165,372). coords(mws1gt_3,44,160,372).  visible(mws1gt_3,372). happensAt(walking(mws1gt_3),372). orientation(mws1gt_6,160,372). coords(mws1gt_6,26,210,372).  visible(mws1gt_6,372). happensAt(walking(mws1gt_6),372).
orientation(mws1gt_3,165,373). coords(mws1gt_3,44,160,373).  visible(mws1gt_3,373). happensAt(walking(mws1gt_3),373). orientation(mws1gt_6,160,373). coords(mws1gt_6,26,211,373).  visible(mws1gt_6,373). happensAt(walking(mws1gt_6),373).
orientation(mws1gt_3,165,374). coords(mws1gt_3,43,160,374).  visible(mws1gt_3,374). happensAt(walking(mws1gt_3),374). orientation(mws1gt_6,160,374). coords(mws1gt_6,25,211,374).  visible(mws1gt_6,374). happensAt(walking(mws1gt_6),374).
orientation(mws1gt_3,165,375). coords(mws1gt_3,42,160,375).  visible(mws1gt_3,375). happensAt(walking(mws1gt_3),375). orientation(mws1gt_6,160,375). coords(mws1gt_6,24,211,375).  visible(mws1gt_6,375). happensAt(walking(mws1gt_6),375).
orientation(mws1gt_3,165,376). coords(mws1gt_3,42,160,376).  visible(mws1gt_3,376). happensAt(walking(mws1gt_3),376). orientation(mws1gt_6,160,376). coords(mws1gt_6,24,212,376).  happensAt(disappear(mws1gt_6),376). happensAt(walking(mws1gt_6),376).
#show holdsAt/2.