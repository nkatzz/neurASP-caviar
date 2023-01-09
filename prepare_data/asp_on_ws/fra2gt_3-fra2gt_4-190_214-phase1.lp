
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

	orientation(fra2gt_3,161,190). coords(fra2gt_3,40,183,190).  visible(fra2gt_3,190). happensAt(walking(fra2gt_3),190). orientation(fra2gt_4,100,190). coords(fra2gt_4,261,285,190).  happensAt(appear(fra2gt_4),190). happensAt(walking(fra2gt_4),190).
orientation(fra2gt_3,161,191). coords(fra2gt_3,41,183,191).  visible(fra2gt_3,191). happensAt(walking(fra2gt_3),191). orientation(fra2gt_4,102,191). coords(fra2gt_4,260,282,191).  visible(fra2gt_4,191). happensAt(walking(fra2gt_4),191).
orientation(fra2gt_3,161,192). coords(fra2gt_3,41,183,192).  visible(fra2gt_3,192). happensAt(walking(fra2gt_3),192). orientation(fra2gt_4,45,192). coords(fra2gt_4,242,279,192).  visible(fra2gt_4,192). happensAt(walking(fra2gt_4),192).
orientation(fra2gt_3,161,193). coords(fra2gt_3,42,182,193).  visible(fra2gt_3,193). happensAt(walking(fra2gt_3),193). orientation(fra2gt_4,45,193). coords(fra2gt_4,240,277,193).  visible(fra2gt_4,193). happensAt(walking(fra2gt_4),193).
orientation(fra2gt_3,161,194). coords(fra2gt_3,42,182,194).  visible(fra2gt_3,194). happensAt(walking(fra2gt_3),194). orientation(fra2gt_4,45,194). coords(fra2gt_4,238,276,194).  visible(fra2gt_4,194). happensAt(walking(fra2gt_4),194).
orientation(fra2gt_3,161,195). coords(fra2gt_3,42,182,195).  visible(fra2gt_3,195). happensAt(walking(fra2gt_3),195). orientation(fra2gt_4,45,195). coords(fra2gt_4,238,275,195).  visible(fra2gt_4,195). happensAt(walking(fra2gt_4),195).
orientation(fra2gt_3,161,196). coords(fra2gt_3,42,181,196).  visible(fra2gt_3,196). happensAt(walking(fra2gt_3),196). orientation(fra2gt_4,45,196). coords(fra2gt_4,238,275,196).  visible(fra2gt_4,196). happensAt(walking(fra2gt_4),196).
orientation(fra2gt_3,161,197). coords(fra2gt_3,42,181,197).  visible(fra2gt_3,197). happensAt(walking(fra2gt_3),197). orientation(fra2gt_4,45,197). coords(fra2gt_4,236,275,197).  visible(fra2gt_4,197). happensAt(walking(fra2gt_4),197).
orientation(fra2gt_3,161,198). coords(fra2gt_3,42,179,198).  visible(fra2gt_3,198). happensAt(walking(fra2gt_3),198). orientation(fra2gt_4,45,198). coords(fra2gt_4,235,275,198).  visible(fra2gt_4,198). happensAt(walking(fra2gt_4),198).
orientation(fra2gt_3,161,199). coords(fra2gt_3,43,178,199).  visible(fra2gt_3,199). happensAt(walking(fra2gt_3),199). orientation(fra2gt_4,37,199). coords(fra2gt_4,234,275,199).  visible(fra2gt_4,199). happensAt(walking(fra2gt_4),199).
orientation(fra2gt_3,161,200). coords(fra2gt_3,43,177,200).  visible(fra2gt_3,200). happensAt(walking(fra2gt_3),200). orientation(fra2gt_4,37,200). coords(fra2gt_4,233,275,200).  visible(fra2gt_4,200). happensAt(walking(fra2gt_4),200).
orientation(fra2gt_3,161,201). coords(fra2gt_3,43,177,201).  visible(fra2gt_3,201). happensAt(walking(fra2gt_3),201). orientation(fra2gt_4,37,201). coords(fra2gt_4,232,275,201).  visible(fra2gt_4,201). happensAt(walking(fra2gt_4),201).
orientation(fra2gt_3,161,202). coords(fra2gt_3,43,177,202).  visible(fra2gt_3,202). happensAt(walking(fra2gt_3),202). orientation(fra2gt_4,37,202). coords(fra2gt_4,231,272,202).  visible(fra2gt_4,202). happensAt(walking(fra2gt_4),202).
orientation(fra2gt_3,161,203). coords(fra2gt_3,44,176,203).  visible(fra2gt_3,203). happensAt(walking(fra2gt_3),203). orientation(fra2gt_4,18,203). coords(fra2gt_4,230,270,203).  visible(fra2gt_4,203). happensAt(walking(fra2gt_4),203).
orientation(fra2gt_3,161,204). coords(fra2gt_3,44,175,204).  visible(fra2gt_3,204). happensAt(walking(fra2gt_3),204). orientation(fra2gt_4,18,204). coords(fra2gt_4,229,265,204).  visible(fra2gt_4,204). happensAt(walking(fra2gt_4),204).
orientation(fra2gt_3,161,205). coords(fra2gt_3,44,174,205).  visible(fra2gt_3,205). happensAt(walking(fra2gt_3),205). orientation(fra2gt_4,18,205). coords(fra2gt_4,229,261,205).  visible(fra2gt_4,205). happensAt(walking(fra2gt_4),205).
orientation(fra2gt_3,161,206). coords(fra2gt_3,44,174,206).  visible(fra2gt_3,206). happensAt(walking(fra2gt_3),206). orientation(fra2gt_4,18,206). coords(fra2gt_4,228,257,206).  visible(fra2gt_4,206). happensAt(walking(fra2gt_4),206).
orientation(fra2gt_3,161,207). coords(fra2gt_3,45,174,207).  visible(fra2gt_3,207). happensAt(walking(fra2gt_3),207). orientation(fra2gt_4,18,207). coords(fra2gt_4,226,256,207).  visible(fra2gt_4,207). happensAt(walking(fra2gt_4),207).
orientation(fra2gt_3,161,208). coords(fra2gt_3,45,173,208).  visible(fra2gt_3,208). happensAt(walking(fra2gt_3),208). orientation(fra2gt_4,18,208). coords(fra2gt_4,225,255,208).  visible(fra2gt_4,208). happensAt(walking(fra2gt_4),208).
orientation(fra2gt_3,161,209). coords(fra2gt_3,45,173,209).  visible(fra2gt_3,209). happensAt(walking(fra2gt_3),209). orientation(fra2gt_4,18,209). coords(fra2gt_4,224,253,209).  visible(fra2gt_4,209). happensAt(walking(fra2gt_4),209).
orientation(fra2gt_3,161,210). coords(fra2gt_3,46,172,210).  visible(fra2gt_3,210). happensAt(walking(fra2gt_3),210). orientation(fra2gt_4,18,210). coords(fra2gt_4,222,251,210).  visible(fra2gt_4,210). happensAt(walking(fra2gt_4),210).
orientation(fra2gt_3,161,211). coords(fra2gt_3,46,171,211).  visible(fra2gt_3,211). happensAt(walking(fra2gt_3),211). orientation(fra2gt_4,18,211). coords(fra2gt_4,220,248,211).  visible(fra2gt_4,211). happensAt(walking(fra2gt_4),211).
orientation(fra2gt_3,161,212). coords(fra2gt_3,46,170,212).  visible(fra2gt_3,212). happensAt(walking(fra2gt_3),212). orientation(fra2gt_4,7,212). coords(fra2gt_4,218,245,212).  visible(fra2gt_4,212). happensAt(walking(fra2gt_4),212).
orientation(fra2gt_3,161,213). coords(fra2gt_3,46,169,213).  visible(fra2gt_3,213). happensAt(walking(fra2gt_3),213). orientation(fra2gt_4,7,213). coords(fra2gt_4,215,243,213).  visible(fra2gt_4,213). happensAt(walking(fra2gt_4),213).
orientation(fra2gt_3,161,214). coords(fra2gt_3,46,168,214).  visible(fra2gt_3,214). happensAt(walking(fra2gt_3),214). orientation(fra2gt_4,7,214). coords(fra2gt_4,212,242,214).  visible(fra2gt_4,214). happensAt(walking(fra2gt_4),214).
#show holdsAt/2.