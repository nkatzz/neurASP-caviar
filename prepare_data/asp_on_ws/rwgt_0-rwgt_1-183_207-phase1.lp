
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

	orientation(rwgt_0,149,183). coords(rwgt_0,170,117,183).  visible(rwgt_0,183). happensAt(inactive(rwgt_0),183). orientation(rwgt_1,135,183). coords(rwgt_1,138,45,183).  visible(rwgt_1,183). happensAt(walking(rwgt_1),183).
orientation(rwgt_0,149,184). coords(rwgt_0,170,117,184).  visible(rwgt_0,184). happensAt(inactive(rwgt_0),184). orientation(rwgt_1,135,184). coords(rwgt_1,138,45,184).  visible(rwgt_1,184). happensAt(walking(rwgt_1),184).
orientation(rwgt_0,149,185). coords(rwgt_0,170,117,185).  visible(rwgt_0,185). happensAt(inactive(rwgt_0),185). orientation(rwgt_1,135,185). coords(rwgt_1,138,45,185).  visible(rwgt_1,185). happensAt(walking(rwgt_1),185).
orientation(rwgt_0,149,186). coords(rwgt_0,170,117,186).  visible(rwgt_0,186). happensAt(inactive(rwgt_0),186). orientation(rwgt_1,135,186). coords(rwgt_1,138,45,186).  visible(rwgt_1,186). happensAt(walking(rwgt_1),186).
orientation(rwgt_0,149,187). coords(rwgt_0,171,117,187).  visible(rwgt_0,187). happensAt(inactive(rwgt_0),187). orientation(rwgt_1,135,187). coords(rwgt_1,138,44,187).  visible(rwgt_1,187). happensAt(walking(rwgt_1),187).
orientation(rwgt_0,149,188). coords(rwgt_0,171,117,188).  visible(rwgt_0,188). happensAt(active(rwgt_0),188). orientation(rwgt_1,135,188). coords(rwgt_1,138,43,188).  visible(rwgt_1,188). happensAt(walking(rwgt_1),188).
orientation(rwgt_0,149,189). coords(rwgt_0,171,117,189).  visible(rwgt_0,189). happensAt(active(rwgt_0),189). orientation(rwgt_1,135,189). coords(rwgt_1,139,43,189).  visible(rwgt_1,189). happensAt(walking(rwgt_1),189).
orientation(rwgt_0,149,190). coords(rwgt_0,171,118,190).  visible(rwgt_0,190). happensAt(active(rwgt_0),190). orientation(rwgt_1,135,190). coords(rwgt_1,139,43,190).  visible(rwgt_1,190). happensAt(walking(rwgt_1),190).
orientation(rwgt_0,149,191). coords(rwgt_0,173,119,191).  visible(rwgt_0,191). happensAt(active(rwgt_0),191). orientation(rwgt_1,135,191). coords(rwgt_1,141,43,191).  visible(rwgt_1,191). happensAt(walking(rwgt_1),191).
orientation(rwgt_0,149,192). coords(rwgt_0,175,121,192).  visible(rwgt_0,192). happensAt(active(rwgt_0),192). orientation(rwgt_1,135,192). coords(rwgt_1,141,42,192).  visible(rwgt_1,192). happensAt(walking(rwgt_1),192).
orientation(rwgt_0,149,193). coords(rwgt_0,175,122,193).  visible(rwgt_0,193). happensAt(active(rwgt_0),193). orientation(rwgt_1,135,193). coords(rwgt_1,141,42,193).  visible(rwgt_1,193). happensAt(walking(rwgt_1),193).
orientation(rwgt_0,149,194). coords(rwgt_0,176,125,194).  visible(rwgt_0,194). happensAt(active(rwgt_0),194). orientation(rwgt_1,135,194). coords(rwgt_1,141,42,194).  visible(rwgt_1,194). happensAt(walking(rwgt_1),194).
orientation(rwgt_0,149,195). coords(rwgt_0,177,127,195).  visible(rwgt_0,195). happensAt(active(rwgt_0),195). orientation(rwgt_1,135,195). coords(rwgt_1,141,42,195).  visible(rwgt_1,195). happensAt(walking(rwgt_1),195).
orientation(rwgt_0,149,196). coords(rwgt_0,179,129,196).  visible(rwgt_0,196). happensAt(active(rwgt_0),196). orientation(rwgt_1,135,196). coords(rwgt_1,142,42,196).  visible(rwgt_1,196). happensAt(walking(rwgt_1),196).
orientation(rwgt_0,149,197). coords(rwgt_0,180,129,197).  visible(rwgt_0,197). happensAt(active(rwgt_0),197). orientation(rwgt_1,135,197). coords(rwgt_1,142,40,197).  visible(rwgt_1,197). happensAt(walking(rwgt_1),197).
orientation(rwgt_0,144,198). coords(rwgt_0,181,133,198).  visible(rwgt_0,198). happensAt(active(rwgt_0),198). orientation(rwgt_1,135,198). coords(rwgt_1,143,39,198).  visible(rwgt_1,198). happensAt(walking(rwgt_1),198).
orientation(rwgt_0,144,199). coords(rwgt_0,183,135,199).  visible(rwgt_0,199). happensAt(active(rwgt_0),199). orientation(rwgt_1,135,199). coords(rwgt_1,143,39,199).  visible(rwgt_1,199). happensAt(walking(rwgt_1),199).
orientation(rwgt_0,144,200). coords(rwgt_0,184,136,200).  visible(rwgt_0,200). happensAt(active(rwgt_0),200). orientation(rwgt_1,135,200). coords(rwgt_1,145,39,200).  visible(rwgt_1,200). happensAt(walking(rwgt_1),200).
orientation(rwgt_0,144,201). coords(rwgt_0,185,137,201).  visible(rwgt_0,201). happensAt(active(rwgt_0),201). orientation(rwgt_1,135,201). coords(rwgt_1,145,39,201).  visible(rwgt_1,201). happensAt(walking(rwgt_1),201).
orientation(rwgt_0,144,202). coords(rwgt_0,186,138,202).  visible(rwgt_0,202). happensAt(active(rwgt_0),202). orientation(rwgt_1,135,202). coords(rwgt_1,146,38,202).  visible(rwgt_1,202). happensAt(walking(rwgt_1),202).
orientation(rwgt_0,144,203). coords(rwgt_0,187,139,203).  visible(rwgt_0,203). happensAt(walking(rwgt_0),203). orientation(rwgt_1,134,203). coords(rwgt_1,147,38,203).  visible(rwgt_1,203). happensAt(walking(rwgt_1),203).
orientation(rwgt_0,144,204). coords(rwgt_0,187,140,204).  visible(rwgt_0,204). happensAt(walking(rwgt_0),204). orientation(rwgt_1,133,204). coords(rwgt_1,149,38,204).  visible(rwgt_1,204). happensAt(walking(rwgt_1),204).
orientation(rwgt_0,144,205). coords(rwgt_0,189,140,205).  visible(rwgt_0,205). happensAt(walking(rwgt_0),205). orientation(rwgt_1,132,205). coords(rwgt_1,152,38,205).  visible(rwgt_1,205). happensAt(walking(rwgt_1),205).
orientation(rwgt_0,144,206). coords(rwgt_0,189,141,206).  visible(rwgt_0,206). happensAt(walking(rwgt_0),206). orientation(rwgt_1,132,206). coords(rwgt_1,152,38,206).  visible(rwgt_1,206). happensAt(walking(rwgt_1),206).
orientation(rwgt_0,144,207). coords(rwgt_0,189,141,207).  visible(rwgt_0,207). happensAt(walking(rwgt_0),207). orientation(rwgt_1,132,207). coords(rwgt_1,153,38,207).  visible(rwgt_1,207). happensAt(walking(rwgt_1),207).
#show holdsAt/2.