
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

	orientation(fcgt_3,164,360). coords(fcgt_3,99,187,360).  visible(fcgt_3,360). happensAt(inactive(fcgt_3),360). orientation(fcgt_4,166,360). coords(fcgt_4,82,206,360).  visible(fcgt_4,360). happensAt(inactive(fcgt_4),360).
orientation(fcgt_3,164,361). coords(fcgt_3,99,187,361).  visible(fcgt_3,361). happensAt(inactive(fcgt_3),361). orientation(fcgt_4,166,361). coords(fcgt_4,82,206,361).  visible(fcgt_4,361). happensAt(inactive(fcgt_4),361).
orientation(fcgt_3,164,362). coords(fcgt_3,99,187,362).  visible(fcgt_3,362). happensAt(inactive(fcgt_3),362). orientation(fcgt_4,166,362). coords(fcgt_4,82,206,362).  visible(fcgt_4,362). happensAt(inactive(fcgt_4),362).
orientation(fcgt_3,164,363). coords(fcgt_3,99,187,363).  visible(fcgt_3,363). happensAt(inactive(fcgt_3),363). orientation(fcgt_4,166,363). coords(fcgt_4,82,206,363).  visible(fcgt_4,363). happensAt(inactive(fcgt_4),363).
orientation(fcgt_3,164,364). coords(fcgt_3,99,187,364).  visible(fcgt_3,364). happensAt(inactive(fcgt_3),364). orientation(fcgt_4,166,364). coords(fcgt_4,82,206,364).  visible(fcgt_4,364). happensAt(inactive(fcgt_4),364).
orientation(fcgt_3,164,365). coords(fcgt_3,99,187,365).  visible(fcgt_3,365). happensAt(inactive(fcgt_3),365). orientation(fcgt_4,166,365). coords(fcgt_4,82,206,365).  visible(fcgt_4,365). happensAt(inactive(fcgt_4),365).
orientation(fcgt_3,164,366). coords(fcgt_3,99,187,366).  visible(fcgt_3,366). happensAt(inactive(fcgt_3),366). orientation(fcgt_4,166,366). coords(fcgt_4,82,206,366).  visible(fcgt_4,366). happensAt(inactive(fcgt_4),366).
orientation(fcgt_3,164,367). coords(fcgt_3,99,187,367).  visible(fcgt_3,367). happensAt(inactive(fcgt_3),367). orientation(fcgt_4,166,367). coords(fcgt_4,82,206,367).  visible(fcgt_4,367). happensAt(inactive(fcgt_4),367).
orientation(fcgt_3,164,368). coords(fcgt_3,99,187,368).  visible(fcgt_3,368). happensAt(inactive(fcgt_3),368). orientation(fcgt_4,166,368). coords(fcgt_4,82,206,368).  visible(fcgt_4,368). happensAt(inactive(fcgt_4),368).
orientation(fcgt_3,164,369). coords(fcgt_3,99,187,369).  visible(fcgt_3,369). happensAt(inactive(fcgt_3),369). orientation(fcgt_4,166,369). coords(fcgt_4,82,206,369).  visible(fcgt_4,369). happensAt(inactive(fcgt_4),369).
orientation(fcgt_3,164,370). coords(fcgt_3,99,187,370).  visible(fcgt_3,370). happensAt(inactive(fcgt_3),370). orientation(fcgt_4,166,370). coords(fcgt_4,82,206,370).  visible(fcgt_4,370). happensAt(inactive(fcgt_4),370).
orientation(fcgt_3,164,371). coords(fcgt_3,99,187,371).  visible(fcgt_3,371). happensAt(inactive(fcgt_3),371). orientation(fcgt_4,166,371). coords(fcgt_4,82,206,371).  visible(fcgt_4,371). happensAt(inactive(fcgt_4),371).
orientation(fcgt_3,164,372). coords(fcgt_3,99,187,372).  visible(fcgt_3,372). happensAt(inactive(fcgt_3),372). orientation(fcgt_4,166,372). coords(fcgt_4,82,206,372).  visible(fcgt_4,372). happensAt(inactive(fcgt_4),372).
orientation(fcgt_3,164,373). coords(fcgt_3,99,187,373).  visible(fcgt_3,373). happensAt(inactive(fcgt_3),373). orientation(fcgt_4,166,373). coords(fcgt_4,82,206,373).  visible(fcgt_4,373). happensAt(inactive(fcgt_4),373).
orientation(fcgt_3,164,374). coords(fcgt_3,99,187,374).  visible(fcgt_3,374). happensAt(inactive(fcgt_3),374). orientation(fcgt_4,166,374). coords(fcgt_4,82,206,374).  visible(fcgt_4,374). happensAt(inactive(fcgt_4),374).
orientation(fcgt_3,164,375). coords(fcgt_3,99,187,375).  visible(fcgt_3,375). happensAt(inactive(fcgt_3),375). orientation(fcgt_4,166,375). coords(fcgt_4,82,206,375).  visible(fcgt_4,375). happensAt(inactive(fcgt_4),375).
orientation(fcgt_3,164,376). coords(fcgt_3,99,187,376).  visible(fcgt_3,376). happensAt(inactive(fcgt_3),376). orientation(fcgt_4,166,376). coords(fcgt_4,82,206,376).  visible(fcgt_4,376). happensAt(inactive(fcgt_4),376).
orientation(fcgt_3,164,377). coords(fcgt_3,99,187,377).  visible(fcgt_3,377). happensAt(inactive(fcgt_3),377). orientation(fcgt_4,166,377). coords(fcgt_4,82,206,377).  visible(fcgt_4,377). happensAt(inactive(fcgt_4),377).
orientation(fcgt_3,164,378). coords(fcgt_3,99,187,378).  visible(fcgt_3,378). happensAt(inactive(fcgt_3),378). orientation(fcgt_4,166,378). coords(fcgt_4,82,206,378).  visible(fcgt_4,378). happensAt(inactive(fcgt_4),378).
orientation(fcgt_3,164,379). coords(fcgt_3,99,187,379).  visible(fcgt_3,379). happensAt(walking(fcgt_3),379). orientation(fcgt_4,166,379). coords(fcgt_4,82,206,379).  visible(fcgt_4,379). happensAt(inactive(fcgt_4),379).
orientation(fcgt_3,164,380). coords(fcgt_3,99,187,380).  visible(fcgt_3,380). happensAt(walking(fcgt_3),380). orientation(fcgt_4,166,380). coords(fcgt_4,82,206,380).  visible(fcgt_4,380). happensAt(inactive(fcgt_4),380).
orientation(fcgt_3,164,381). coords(fcgt_3,99,187,381).  visible(fcgt_3,381). happensAt(walking(fcgt_3),381). orientation(fcgt_4,166,381). coords(fcgt_4,82,206,381).  visible(fcgt_4,381). happensAt(inactive(fcgt_4),381).
orientation(fcgt_3,164,382). coords(fcgt_3,99,187,382).  visible(fcgt_3,382). happensAt(walking(fcgt_3),382). orientation(fcgt_4,166,382). coords(fcgt_4,82,206,382).  visible(fcgt_4,382). happensAt(inactive(fcgt_4),382).
orientation(fcgt_3,164,383). coords(fcgt_3,99,187,383).  visible(fcgt_3,383). happensAt(walking(fcgt_3),383). orientation(fcgt_4,166,383). coords(fcgt_4,82,206,383).  visible(fcgt_4,383). happensAt(inactive(fcgt_4),383).
orientation(fcgt_3,164,384). coords(fcgt_3,99,187,384).  visible(fcgt_3,384). happensAt(walking(fcgt_3),384). orientation(fcgt_4,166,384). coords(fcgt_4,82,206,384).  visible(fcgt_4,384). happensAt(inactive(fcgt_4),384).
#show holdsAt/2.