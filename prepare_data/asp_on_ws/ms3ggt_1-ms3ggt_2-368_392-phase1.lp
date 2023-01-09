
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

	orientation(ms3ggt_1,156,368). coords(ms3ggt_1,126,158,368).  visible(ms3ggt_1,368). happensAt(walking(ms3ggt_1),368). orientation(ms3ggt_2,173,368). coords(ms3ggt_2,180,218,368).  visible(ms3ggt_2,368). happensAt(walking(ms3ggt_2),368).
orientation(ms3ggt_1,156,369). coords(ms3ggt_1,127,158,369).  visible(ms3ggt_1,369). happensAt(walking(ms3ggt_1),369). orientation(ms3ggt_2,173,369). coords(ms3ggt_2,178,217,369).  visible(ms3ggt_2,369). happensAt(walking(ms3ggt_2),369).
orientation(ms3ggt_1,156,370). coords(ms3ggt_1,127,158,370).  visible(ms3ggt_1,370). happensAt(walking(ms3ggt_1),370). orientation(ms3ggt_2,173,370). coords(ms3ggt_2,175,217,370).  visible(ms3ggt_2,370). happensAt(walking(ms3ggt_2),370).
orientation(ms3ggt_1,156,371). coords(ms3ggt_1,128,158,371).  visible(ms3ggt_1,371). happensAt(walking(ms3ggt_1),371). orientation(ms3ggt_2,173,371). coords(ms3ggt_2,174,217,371).  visible(ms3ggt_2,371). happensAt(walking(ms3ggt_2),371).
orientation(ms3ggt_1,156,372). coords(ms3ggt_1,128,158,372).  visible(ms3ggt_1,372). happensAt(walking(ms3ggt_1),372). orientation(ms3ggt_2,173,372). coords(ms3ggt_2,171,216,372).  visible(ms3ggt_2,372). happensAt(walking(ms3ggt_2),372).
orientation(ms3ggt_1,156,373). coords(ms3ggt_1,129,158,373).  visible(ms3ggt_1,373). happensAt(walking(ms3ggt_1),373). orientation(ms3ggt_2,173,373). coords(ms3ggt_2,170,216,373).  visible(ms3ggt_2,373). happensAt(walking(ms3ggt_2),373).
orientation(ms3ggt_1,156,374). coords(ms3ggt_1,130,158,374).  visible(ms3ggt_1,374). happensAt(walking(ms3ggt_1),374). orientation(ms3ggt_2,173,374). coords(ms3ggt_2,169,215,374).  visible(ms3ggt_2,374). happensAt(walking(ms3ggt_2),374).
orientation(ms3ggt_1,156,375). coords(ms3ggt_1,130,158,375).  visible(ms3ggt_1,375). happensAt(walking(ms3ggt_1),375). orientation(ms3ggt_2,173,375). coords(ms3ggt_2,169,215,375).  visible(ms3ggt_2,375). happensAt(walking(ms3ggt_2),375).
orientation(ms3ggt_1,156,376). coords(ms3ggt_1,130,158,376).  visible(ms3ggt_1,376). happensAt(walking(ms3ggt_1),376). orientation(ms3ggt_2,173,376). coords(ms3ggt_2,168,212,376).  visible(ms3ggt_2,376). happensAt(walking(ms3ggt_2),376).
orientation(ms3ggt_1,156,377). coords(ms3ggt_1,132,158,377).  visible(ms3ggt_1,377). happensAt(walking(ms3ggt_1),377). orientation(ms3ggt_2,173,377). coords(ms3ggt_2,168,212,377).  visible(ms3ggt_2,377). happensAt(walking(ms3ggt_2),377).
orientation(ms3ggt_1,156,378). coords(ms3ggt_1,134,158,378).  visible(ms3ggt_1,378). happensAt(walking(ms3ggt_1),378). orientation(ms3ggt_2,173,378). coords(ms3ggt_2,167,211,378).  visible(ms3ggt_2,378). happensAt(walking(ms3ggt_2),378).
orientation(ms3ggt_1,162,379). coords(ms3ggt_1,136,158,379).  visible(ms3ggt_1,379). happensAt(walking(ms3ggt_1),379). orientation(ms3ggt_2,173,379). coords(ms3ggt_2,166,210,379).  visible(ms3ggt_2,379). happensAt(walking(ms3ggt_2),379).
orientation(ms3ggt_1,162,380). coords(ms3ggt_1,138,158,380).  visible(ms3ggt_1,380). happensAt(walking(ms3ggt_1),380). orientation(ms3ggt_2,173,380). coords(ms3ggt_2,166,210,380).  visible(ms3ggt_2,380). happensAt(walking(ms3ggt_2),380).
orientation(ms3ggt_1,162,381). coords(ms3ggt_1,138,158,381).  visible(ms3ggt_1,381). happensAt(walking(ms3ggt_1),381). orientation(ms3ggt_2,173,381). coords(ms3ggt_2,165,209,381).  visible(ms3ggt_2,381). happensAt(walking(ms3ggt_2),381).
orientation(ms3ggt_1,162,382). coords(ms3ggt_1,140,158,382).  visible(ms3ggt_1,382). happensAt(walking(ms3ggt_1),382). orientation(ms3ggt_2,165,382). coords(ms3ggt_2,163,208,382).  visible(ms3ggt_2,382). happensAt(walking(ms3ggt_2),382).
orientation(ms3ggt_1,162,383). coords(ms3ggt_1,141,158,383).  visible(ms3ggt_1,383). happensAt(walking(ms3ggt_1),383). orientation(ms3ggt_2,165,383). coords(ms3ggt_2,159,206,383).  visible(ms3ggt_2,383). happensAt(walking(ms3ggt_2),383).
orientation(ms3ggt_1,162,384). coords(ms3ggt_1,142,158,384).  visible(ms3ggt_1,384). happensAt(walking(ms3ggt_1),384). orientation(ms3ggt_2,165,384). coords(ms3ggt_2,159,204,384).  visible(ms3ggt_2,384). happensAt(walking(ms3ggt_2),384).
orientation(ms3ggt_1,162,385). coords(ms3ggt_1,142,157,385).  visible(ms3ggt_1,385). happensAt(walking(ms3ggt_1),385). orientation(ms3ggt_2,168,385). coords(ms3ggt_2,155,203,385).  visible(ms3ggt_2,385). happensAt(walking(ms3ggt_2),385).
orientation(ms3ggt_1,162,386). coords(ms3ggt_1,142,157,386).  visible(ms3ggt_1,386). happensAt(walking(ms3ggt_1),386). orientation(ms3ggt_2,168,386). coords(ms3ggt_2,154,203,386).  visible(ms3ggt_2,386). happensAt(walking(ms3ggt_2),386).
orientation(ms3ggt_1,162,387). coords(ms3ggt_1,142,157,387).  visible(ms3ggt_1,387). happensAt(walking(ms3ggt_1),387). orientation(ms3ggt_2,168,387). coords(ms3ggt_2,153,201,387).  visible(ms3ggt_2,387). happensAt(walking(ms3ggt_2),387).
orientation(ms3ggt_1,162,388). coords(ms3ggt_1,143,157,388).  visible(ms3ggt_1,388). happensAt(walking(ms3ggt_1),388). orientation(ms3ggt_2,168,388). coords(ms3ggt_2,153,201,388).  visible(ms3ggt_2,388). happensAt(walking(ms3ggt_2),388).
orientation(ms3ggt_1,162,389). coords(ms3ggt_1,143,157,389).  visible(ms3ggt_1,389). happensAt(walking(ms3ggt_1),389). orientation(ms3ggt_2,168,389). coords(ms3ggt_2,153,201,389).  visible(ms3ggt_2,389). happensAt(walking(ms3ggt_2),389).
orientation(ms3ggt_1,162,390). coords(ms3ggt_1,143,157,390).  visible(ms3ggt_1,390). happensAt(walking(ms3ggt_1),390). orientation(ms3ggt_2,168,390). coords(ms3ggt_2,152,201,390).  visible(ms3ggt_2,390). happensAt(walking(ms3ggt_2),390).
orientation(ms3ggt_1,162,391). coords(ms3ggt_1,144,157,391).  visible(ms3ggt_1,391). happensAt(walking(ms3ggt_1),391). orientation(ms3ggt_2,168,391). coords(ms3ggt_2,152,200,391).  visible(ms3ggt_2,391). happensAt(walking(ms3ggt_2),391).
orientation(ms3ggt_1,162,392). coords(ms3ggt_1,144,158,392).  visible(ms3ggt_1,392). happensAt(walking(ms3ggt_1),392). orientation(ms3ggt_2,168,392). coords(ms3ggt_2,151,200,392).  visible(ms3ggt_2,392). happensAt(walking(ms3ggt_2),392).
#show holdsAt/2.