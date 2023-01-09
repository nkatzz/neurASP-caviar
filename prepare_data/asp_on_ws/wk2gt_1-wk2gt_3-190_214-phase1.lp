
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

	orientation(wk2gt_1,129,190). coords(wk2gt_1,115,53,190).  visible(wk2gt_1,190). happensAt(inactive(wk2gt_1),190). orientation(wk2gt_3,157,190). coords(wk2gt_3,237,216,190).  visible(wk2gt_3,190). happensAt(walking(wk2gt_3),190).
orientation(wk2gt_1,129,191). coords(wk2gt_1,115,53,191).  visible(wk2gt_1,191). happensAt(inactive(wk2gt_1),191). orientation(wk2gt_3,162,191). coords(wk2gt_3,239,220,191).  visible(wk2gt_3,191). happensAt(walking(wk2gt_3),191).
orientation(wk2gt_1,129,192). coords(wk2gt_1,115,53,192).  visible(wk2gt_1,192). happensAt(inactive(wk2gt_1),192). orientation(wk2gt_3,169,192). coords(wk2gt_3,242,224,192).  visible(wk2gt_3,192). happensAt(walking(wk2gt_3),192).
orientation(wk2gt_1,129,193). coords(wk2gt_1,115,53,193).  visible(wk2gt_1,193). happensAt(inactive(wk2gt_1),193). orientation(wk2gt_3,174,193). coords(wk2gt_3,245,226,193).  visible(wk2gt_3,193). happensAt(walking(wk2gt_3),193).
orientation(wk2gt_1,129,194). coords(wk2gt_1,115,53,194).  visible(wk2gt_1,194). happensAt(inactive(wk2gt_1),194). orientation(wk2gt_3,170,194). coords(wk2gt_3,248,228,194).  visible(wk2gt_3,194). happensAt(walking(wk2gt_3),194).
orientation(wk2gt_1,129,195). coords(wk2gt_1,115,53,195).  visible(wk2gt_1,195). happensAt(inactive(wk2gt_1),195). orientation(wk2gt_3,9,195). coords(wk2gt_3,250,228,195).  visible(wk2gt_3,195). happensAt(walking(wk2gt_3),195).
orientation(wk2gt_1,129,196). coords(wk2gt_1,115,53,196).  visible(wk2gt_1,196). happensAt(inactive(wk2gt_1),196). orientation(wk2gt_3,29,196). coords(wk2gt_3,250,230,196).  visible(wk2gt_3,196). happensAt(walking(wk2gt_3),196).
orientation(wk2gt_1,129,197). coords(wk2gt_1,115,53,197).  visible(wk2gt_1,197). happensAt(inactive(wk2gt_1),197). orientation(wk2gt_3,40,197). coords(wk2gt_3,251,231,197).  visible(wk2gt_3,197). happensAt(walking(wk2gt_3),197).
orientation(wk2gt_1,129,198). coords(wk2gt_1,115,53,198).  visible(wk2gt_1,198). happensAt(inactive(wk2gt_1),198). orientation(wk2gt_3,37,198). coords(wk2gt_3,253,233,198).  visible(wk2gt_3,198). happensAt(walking(wk2gt_3),198).
orientation(wk2gt_1,129,199). coords(wk2gt_1,115,53,199).  visible(wk2gt_1,199). happensAt(inactive(wk2gt_1),199). orientation(wk2gt_3,38,199). coords(wk2gt_3,255,237,199).  visible(wk2gt_3,199). happensAt(walking(wk2gt_3),199).
orientation(wk2gt_1,129,200). coords(wk2gt_1,115,53,200).  visible(wk2gt_1,200). happensAt(inactive(wk2gt_1),200). orientation(wk2gt_3,38,200). coords(wk2gt_3,258,241,200).  visible(wk2gt_3,200). happensAt(walking(wk2gt_3),200).
orientation(wk2gt_1,129,201). coords(wk2gt_1,115,53,201).  visible(wk2gt_1,201). happensAt(inactive(wk2gt_1),201). orientation(wk2gt_3,38,201). coords(wk2gt_3,262,245,201).  visible(wk2gt_3,201). happensAt(walking(wk2gt_3),201).
orientation(wk2gt_1,129,202). coords(wk2gt_1,115,53,202).  visible(wk2gt_1,202). happensAt(inactive(wk2gt_1),202). orientation(wk2gt_3,38,202). coords(wk2gt_3,265,247,202).  visible(wk2gt_3,202). happensAt(walking(wk2gt_3),202).
orientation(wk2gt_1,129,203). coords(wk2gt_1,115,53,203).  visible(wk2gt_1,203). happensAt(inactive(wk2gt_1),203). orientation(wk2gt_3,38,203). coords(wk2gt_3,269,249,203).  visible(wk2gt_3,203). happensAt(walking(wk2gt_3),203).
orientation(wk2gt_1,129,204). coords(wk2gt_1,115,53,204).  visible(wk2gt_1,204). happensAt(inactive(wk2gt_1),204). orientation(wk2gt_3,38,204). coords(wk2gt_3,272,252,204).  visible(wk2gt_3,204). happensAt(walking(wk2gt_3),204).
orientation(wk2gt_1,129,205). coords(wk2gt_1,115,53,205).  visible(wk2gt_1,205). happensAt(inactive(wk2gt_1),205). orientation(wk2gt_3,38,205). coords(wk2gt_3,275,254,205).  visible(wk2gt_3,205). happensAt(walking(wk2gt_3),205).
orientation(wk2gt_1,129,206). coords(wk2gt_1,115,53,206).  visible(wk2gt_1,206). happensAt(inactive(wk2gt_1),206). orientation(wk2gt_3,38,206). coords(wk2gt_3,277,257,206).  visible(wk2gt_3,206). happensAt(walking(wk2gt_3),206).
orientation(wk2gt_1,129,207). coords(wk2gt_1,115,53,207).  visible(wk2gt_1,207). happensAt(inactive(wk2gt_1),207). orientation(wk2gt_3,38,207). coords(wk2gt_3,279,258,207).  visible(wk2gt_3,207). happensAt(walking(wk2gt_3),207).
orientation(wk2gt_1,129,208). coords(wk2gt_1,115,53,208).  visible(wk2gt_1,208). happensAt(inactive(wk2gt_1),208). orientation(wk2gt_3,38,208). coords(wk2gt_3,280,260,208).  visible(wk2gt_3,208). happensAt(walking(wk2gt_3),208).
orientation(wk2gt_1,129,209). coords(wk2gt_1,115,53,209).  visible(wk2gt_1,209). happensAt(inactive(wk2gt_1),209). orientation(wk2gt_3,38,209). coords(wk2gt_3,281,262,209).  visible(wk2gt_3,209). happensAt(walking(wk2gt_3),209).
orientation(wk2gt_1,129,210). coords(wk2gt_1,115,53,210).  visible(wk2gt_1,210). happensAt(inactive(wk2gt_1),210). orientation(wk2gt_3,38,210). coords(wk2gt_3,282,265,210).  visible(wk2gt_3,210). happensAt(walking(wk2gt_3),210).
orientation(wk2gt_1,129,211). coords(wk2gt_1,115,53,211).  visible(wk2gt_1,211). happensAt(inactive(wk2gt_1),211). orientation(wk2gt_3,46,211). coords(wk2gt_3,282,267,211).  visible(wk2gt_3,211). happensAt(walking(wk2gt_3),211).
orientation(wk2gt_1,129,212). coords(wk2gt_1,115,53,212).  visible(wk2gt_1,212). happensAt(inactive(wk2gt_1),212). orientation(wk2gt_3,48,212). coords(wk2gt_3,284,270,212).  visible(wk2gt_3,212). happensAt(walking(wk2gt_3),212).
orientation(wk2gt_1,129,213). coords(wk2gt_1,115,53,213).  visible(wk2gt_1,213). happensAt(inactive(wk2gt_1),213). orientation(wk2gt_3,48,213). coords(wk2gt_3,287,272,213).  visible(wk2gt_3,213). happensAt(walking(wk2gt_3),213).
orientation(wk2gt_1,129,214). coords(wk2gt_1,115,53,214).  visible(wk2gt_1,214). happensAt(inactive(wk2gt_1),214). orientation(wk2gt_3,48,214). coords(wk2gt_3,288,273,214).  visible(wk2gt_3,214). happensAt(walking(wk2gt_3),214).
#show holdsAt/2.