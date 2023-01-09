
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

	orientation(ricgt_3,9,183). coords(ricgt_3,258,225,183).  visible(ricgt_3,183). happensAt(walking(ricgt_3),183). orientation(ricgt_4,172,183). coords(ricgt_4,24,142,183).  visible(ricgt_4,183). happensAt(walking(ricgt_4),183).
orientation(ricgt_3,12,184). coords(ricgt_3,259,225,184).  visible(ricgt_3,184). happensAt(walking(ricgt_3),184). orientation(ricgt_4,172,184). coords(ricgt_4,24,142,184).  visible(ricgt_4,184). happensAt(walking(ricgt_4),184).
orientation(ricgt_3,23,185). coords(ricgt_3,260,226,185).  visible(ricgt_3,185). happensAt(walking(ricgt_3),185). orientation(ricgt_4,172,185). coords(ricgt_4,24,143,185).  visible(ricgt_4,185). happensAt(walking(ricgt_4),185).
orientation(ricgt_3,31,186). coords(ricgt_3,261,228,186).  visible(ricgt_3,186). happensAt(walking(ricgt_3),186). orientation(ricgt_4,172,186). coords(ricgt_4,24,143,186).  visible(ricgt_4,186). happensAt(walking(ricgt_4),186).
orientation(ricgt_3,37,187). coords(ricgt_3,262,230,187).  visible(ricgt_3,187). happensAt(walking(ricgt_3),187). orientation(ricgt_4,172,187). coords(ricgt_4,24,143,187).  visible(ricgt_4,187). happensAt(walking(ricgt_4),187).
orientation(ricgt_3,40,188). coords(ricgt_3,263,234,188).  visible(ricgt_3,188). happensAt(walking(ricgt_3),188). orientation(ricgt_4,172,188). coords(ricgt_4,23,143,188).  visible(ricgt_4,188). happensAt(walking(ricgt_4),188).
orientation(ricgt_3,42,189). coords(ricgt_3,265,237,189).  visible(ricgt_3,189). happensAt(walking(ricgt_3),189). orientation(ricgt_4,172,189). coords(ricgt_4,23,143,189).  visible(ricgt_4,189). happensAt(walking(ricgt_4),189).
orientation(ricgt_3,38,190). coords(ricgt_3,268,240,190).  visible(ricgt_3,190). happensAt(walking(ricgt_3),190). orientation(ricgt_4,172,190). coords(ricgt_4,23,144,190).  visible(ricgt_4,190). happensAt(walking(ricgt_4),190).
orientation(ricgt_3,34,191). coords(ricgt_3,273,242,191).  visible(ricgt_3,191). happensAt(walking(ricgt_3),191). orientation(ricgt_4,172,191). coords(ricgt_4,23,145,191).  visible(ricgt_4,191). happensAt(walking(ricgt_4),191).
orientation(ricgt_3,28,192). coords(ricgt_3,277,244,192).  visible(ricgt_3,192). happensAt(walking(ricgt_3),192). orientation(ricgt_4,172,192). coords(ricgt_4,23,145,192).  visible(ricgt_4,192). happensAt(walking(ricgt_4),192).
orientation(ricgt_3,34,193). coords(ricgt_3,280,246,193).  visible(ricgt_3,193). happensAt(walking(ricgt_3),193). orientation(ricgt_4,172,193). coords(ricgt_4,23,146,193).  visible(ricgt_4,193). happensAt(walking(ricgt_4),193).
orientation(ricgt_3,32,194). coords(ricgt_3,283,248,194).  visible(ricgt_3,194). happensAt(walking(ricgt_3),194). orientation(ricgt_4,172,194). coords(ricgt_4,23,146,194).  visible(ricgt_4,194). happensAt(walking(ricgt_4),194).
orientation(ricgt_3,37,195). coords(ricgt_3,285,249,195).  visible(ricgt_3,195). happensAt(walking(ricgt_3),195). orientation(ricgt_4,172,195). coords(ricgt_4,23,147,195).  visible(ricgt_4,195). happensAt(walking(ricgt_4),195).
orientation(ricgt_3,41,196). coords(ricgt_3,287,252,196).  visible(ricgt_3,196). happensAt(walking(ricgt_3),196). orientation(ricgt_4,172,196). coords(ricgt_4,23,147,196).  visible(ricgt_4,196). happensAt(walking(ricgt_4),196).
orientation(ricgt_3,45,197). coords(ricgt_3,288,253,197).  visible(ricgt_3,197). happensAt(walking(ricgt_3),197). orientation(ricgt_4,172,197). coords(ricgt_4,23,148,197).  visible(ricgt_4,197). happensAt(walking(ricgt_4),197).
orientation(ricgt_3,49,198). coords(ricgt_3,289,255,198).  visible(ricgt_3,198). happensAt(walking(ricgt_3),198). orientation(ricgt_4,172,198). coords(ricgt_4,23,148,198).  visible(ricgt_4,198). happensAt(walking(ricgt_4),198).
orientation(ricgt_3,52,199). coords(ricgt_3,290,258,199).  visible(ricgt_3,199). happensAt(walking(ricgt_3),199). orientation(ricgt_4,172,199). coords(ricgt_4,23,149,199).  visible(ricgt_4,199). happensAt(walking(ricgt_4),199).
orientation(ricgt_3,56,200). coords(ricgt_3,291,261,200).  visible(ricgt_3,200). happensAt(walking(ricgt_3),200). orientation(ricgt_4,172,200). coords(ricgt_4,23,149,200).  visible(ricgt_4,200). happensAt(walking(ricgt_4),200).
orientation(ricgt_3,60,201). coords(ricgt_3,292,264,201).  visible(ricgt_3,201). happensAt(walking(ricgt_3),201). orientation(ricgt_4,172,201). coords(ricgt_4,23,150,201).  visible(ricgt_4,201). happensAt(walking(ricgt_4),201).
orientation(ricgt_3,63,202). coords(ricgt_3,293,268,202).  visible(ricgt_3,202). happensAt(walking(ricgt_3),202). orientation(ricgt_4,172,202). coords(ricgt_4,23,150,202).  visible(ricgt_4,202). happensAt(walking(ricgt_4),202).
orientation(ricgt_3,64,203). coords(ricgt_3,294,269,203).  visible(ricgt_3,203). happensAt(walking(ricgt_3),203). orientation(ricgt_4,172,203). coords(ricgt_4,23,151,203).  visible(ricgt_4,203). happensAt(walking(ricgt_4),203).
orientation(ricgt_3,66,204). coords(ricgt_3,296,269,204).  visible(ricgt_3,204). happensAt(walking(ricgt_3),204). orientation(ricgt_4,172,204). coords(ricgt_4,23,152,204).  visible(ricgt_4,204). happensAt(walking(ricgt_4),204).
orientation(ricgt_3,68,205). coords(ricgt_3,297,269,205).  visible(ricgt_3,205). happensAt(walking(ricgt_3),205). orientation(ricgt_4,172,205). coords(ricgt_4,24,152,205).  visible(ricgt_4,205). happensAt(walking(ricgt_4),205).
orientation(ricgt_3,70,206). coords(ricgt_3,298,269,206).  visible(ricgt_3,206). happensAt(walking(ricgt_3),206). orientation(ricgt_4,172,206). coords(ricgt_4,24,152,206).  visible(ricgt_4,206). happensAt(walking(ricgt_4),206).
orientation(ricgt_3,71,207). coords(ricgt_3,299,269,207).  visible(ricgt_3,207). happensAt(walking(ricgt_3),207). orientation(ricgt_4,172,207). coords(ricgt_4,23,153,207).  visible(ricgt_4,207). happensAt(walking(ricgt_4),207).
#show holdsAt/2.