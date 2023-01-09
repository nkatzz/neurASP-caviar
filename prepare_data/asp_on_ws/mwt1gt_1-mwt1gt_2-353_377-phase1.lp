
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

	orientation(mwt1gt_1,160,353). coords(mwt1gt_1,115,147,353).  visible(mwt1gt_1,353). happensAt(active(mwt1gt_1),353). orientation(mwt1gt_2,140,353). coords(mwt1gt_2,121,125,353).  visible(mwt1gt_2,353). happensAt(active(mwt1gt_2),353).
orientation(mwt1gt_1,160,354). coords(mwt1gt_1,115,147,354).  visible(mwt1gt_1,354). happensAt(active(mwt1gt_1),354). orientation(mwt1gt_2,140,354). coords(mwt1gt_2,121,125,354).  visible(mwt1gt_2,354). happensAt(walking(mwt1gt_2),354).
orientation(mwt1gt_1,160,355). coords(mwt1gt_1,111,147,355).  visible(mwt1gt_1,355). happensAt(active(mwt1gt_1),355). orientation(mwt1gt_2,140,355). coords(mwt1gt_2,118,124,355).  visible(mwt1gt_2,355). happensAt(walking(mwt1gt_2),355).
orientation(mwt1gt_1,160,356). coords(mwt1gt_1,111,147,356).  visible(mwt1gt_1,356). happensAt(walking(mwt1gt_1),356). orientation(mwt1gt_2,140,356). coords(mwt1gt_2,118,124,356).  visible(mwt1gt_2,356). happensAt(walking(mwt1gt_2),356).
orientation(mwt1gt_1,160,357). coords(mwt1gt_1,111,147,357).  visible(mwt1gt_1,357). happensAt(walking(mwt1gt_1),357). orientation(mwt1gt_2,140,357). coords(mwt1gt_2,117,124,357).  visible(mwt1gt_2,357). happensAt(walking(mwt1gt_2),357).
orientation(mwt1gt_1,160,358). coords(mwt1gt_1,111,146,358).  visible(mwt1gt_1,358). happensAt(walking(mwt1gt_1),358). orientation(mwt1gt_2,140,358). coords(mwt1gt_2,117,124,358).  visible(mwt1gt_2,358). happensAt(walking(mwt1gt_2),358).
orientation(mwt1gt_1,160,359). coords(mwt1gt_1,111,146,359).  visible(mwt1gt_1,359). happensAt(walking(mwt1gt_1),359). orientation(mwt1gt_2,140,359). coords(mwt1gt_2,117,124,359).  visible(mwt1gt_2,359). happensAt(walking(mwt1gt_2),359).
orientation(mwt1gt_1,160,360). coords(mwt1gt_1,111,146,360).  visible(mwt1gt_1,360). happensAt(walking(mwt1gt_1),360). orientation(mwt1gt_2,140,360). coords(mwt1gt_2,117,124,360).  visible(mwt1gt_2,360). happensAt(walking(mwt1gt_2),360).
orientation(mwt1gt_1,160,361). coords(mwt1gt_1,111,146,361).  visible(mwt1gt_1,361). happensAt(walking(mwt1gt_1),361). orientation(mwt1gt_2,140,361). coords(mwt1gt_2,117,124,361).  visible(mwt1gt_2,361). happensAt(walking(mwt1gt_2),361).
orientation(mwt1gt_1,160,362). coords(mwt1gt_1,111,145,362).  visible(mwt1gt_1,362). happensAt(walking(mwt1gt_1),362). orientation(mwt1gt_2,140,362). coords(mwt1gt_2,117,124,362).  visible(mwt1gt_2,362). happensAt(walking(mwt1gt_2),362).
orientation(mwt1gt_1,160,363). coords(mwt1gt_1,111,145,363).  visible(mwt1gt_1,363). happensAt(walking(mwt1gt_1),363). orientation(mwt1gt_2,140,363). coords(mwt1gt_2,116,124,363).  visible(mwt1gt_2,363). happensAt(walking(mwt1gt_2),363).
orientation(mwt1gt_1,160,364). coords(mwt1gt_1,109,144,364).  visible(mwt1gt_1,364). happensAt(walking(mwt1gt_1),364). orientation(mwt1gt_2,140,364). coords(mwt1gt_2,116,124,364).  visible(mwt1gt_2,364). happensAt(walking(mwt1gt_2),364).
orientation(mwt1gt_1,160,365). coords(mwt1gt_1,109,144,365).  visible(mwt1gt_1,365). happensAt(walking(mwt1gt_1),365). orientation(mwt1gt_2,140,365). coords(mwt1gt_2,116,124,365).  visible(mwt1gt_2,365). happensAt(walking(mwt1gt_2),365).
orientation(mwt1gt_1,160,366). coords(mwt1gt_1,107,141,366).  visible(mwt1gt_1,366). happensAt(walking(mwt1gt_1),366). orientation(mwt1gt_2,140,366). coords(mwt1gt_2,113,115,366).  visible(mwt1gt_2,366). happensAt(walking(mwt1gt_2),366).
orientation(mwt1gt_1,160,367). coords(mwt1gt_1,106,140,367).  visible(mwt1gt_1,367). happensAt(walking(mwt1gt_1),367). orientation(mwt1gt_2,140,367). coords(mwt1gt_2,113,115,367).  visible(mwt1gt_2,367). happensAt(walking(mwt1gt_2),367).
orientation(mwt1gt_1,160,368). coords(mwt1gt_1,106,139,368).  visible(mwt1gt_1,368). happensAt(walking(mwt1gt_1),368). orientation(mwt1gt_2,140,368). coords(mwt1gt_2,110,115,368).  visible(mwt1gt_2,368). happensAt(walking(mwt1gt_2),368).
orientation(mwt1gt_1,160,369). coords(mwt1gt_1,106,139,369).  visible(mwt1gt_1,369). happensAt(walking(mwt1gt_1),369). orientation(mwt1gt_2,140,369). coords(mwt1gt_2,110,115,369).  visible(mwt1gt_2,369). happensAt(walking(mwt1gt_2),369).
orientation(mwt1gt_1,160,370). coords(mwt1gt_1,105,136,370).  visible(mwt1gt_1,370). happensAt(walking(mwt1gt_1),370). orientation(mwt1gt_2,140,370). coords(mwt1gt_2,110,115,370).  visible(mwt1gt_2,370). happensAt(walking(mwt1gt_2),370).
orientation(mwt1gt_1,160,371). coords(mwt1gt_1,105,136,371).  visible(mwt1gt_1,371). happensAt(walking(mwt1gt_1),371). orientation(mwt1gt_2,140,371). coords(mwt1gt_2,110,115,371).  visible(mwt1gt_2,371). happensAt(walking(mwt1gt_2),371).
orientation(mwt1gt_1,160,372). coords(mwt1gt_1,105,135,372).  visible(mwt1gt_1,372). happensAt(walking(mwt1gt_1),372). orientation(mwt1gt_2,140,372). coords(mwt1gt_2,107,115,372).  visible(mwt1gt_2,372). happensAt(walking(mwt1gt_2),372).
orientation(mwt1gt_1,160,373). coords(mwt1gt_1,105,135,373).  visible(mwt1gt_1,373). happensAt(walking(mwt1gt_1),373). orientation(mwt1gt_2,140,373). coords(mwt1gt_2,107,115,373).  visible(mwt1gt_2,373). happensAt(walking(mwt1gt_2),373).
orientation(mwt1gt_1,160,374). coords(mwt1gt_1,105,135,374).  visible(mwt1gt_1,374). happensAt(walking(mwt1gt_1),374). orientation(mwt1gt_2,140,374). coords(mwt1gt_2,107,115,374).  visible(mwt1gt_2,374). happensAt(walking(mwt1gt_2),374).
orientation(mwt1gt_1,160,375). coords(mwt1gt_1,105,135,375).  visible(mwt1gt_1,375). happensAt(walking(mwt1gt_1),375). orientation(mwt1gt_2,140,375). coords(mwt1gt_2,106,115,375).  visible(mwt1gt_2,375). happensAt(walking(mwt1gt_2),375).
orientation(mwt1gt_1,160,376). coords(mwt1gt_1,105,135,376).  visible(mwt1gt_1,376). happensAt(walking(mwt1gt_1),376). orientation(mwt1gt_2,140,376). coords(mwt1gt_2,105,115,376).  visible(mwt1gt_2,376). happensAt(walking(mwt1gt_2),376).
orientation(mwt1gt_1,160,377). coords(mwt1gt_1,105,135,377).  visible(mwt1gt_1,377). happensAt(walking(mwt1gt_1),377). orientation(mwt1gt_2,140,377). coords(mwt1gt_2,105,114,377).  visible(mwt1gt_2,377). happensAt(walking(mwt1gt_2),377).
#show holdsAt/2.