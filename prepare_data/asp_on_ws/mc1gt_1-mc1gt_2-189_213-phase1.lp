
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

	orientation(mc1gt_1,141,189). coords(mc1gt_1,151,96,189).  visible(mc1gt_1,189). happensAt(walking(mc1gt_1),189). orientation(mc1gt_2,147,189). coords(mc1gt_2,95,115,189).  visible(mc1gt_2,189). happensAt(walking(mc1gt_2),189).
orientation(mc1gt_1,139,190). coords(mc1gt_1,151,97,190).  visible(mc1gt_1,190). happensAt(walking(mc1gt_1),190). orientation(mc1gt_2,148,190). coords(mc1gt_2,97,115,190).  visible(mc1gt_2,190). happensAt(walking(mc1gt_2),190).
orientation(mc1gt_1,139,191). coords(mc1gt_1,152,97,191).  visible(mc1gt_1,191). happensAt(walking(mc1gt_1),191). orientation(mc1gt_2,149,191). coords(mc1gt_2,98,116,191).  visible(mc1gt_2,191). happensAt(walking(mc1gt_2),191).
orientation(mc1gt_1,140,192). coords(mc1gt_1,152,97,192).  visible(mc1gt_1,192). happensAt(walking(mc1gt_1),192). orientation(mc1gt_2,149,192). coords(mc1gt_2,99,116,192).  visible(mc1gt_2,192). happensAt(walking(mc1gt_2),192).
orientation(mc1gt_1,140,193). coords(mc1gt_1,152,97,193).  visible(mc1gt_1,193). happensAt(walking(mc1gt_1),193). orientation(mc1gt_2,149,193). coords(mc1gt_2,100,116,193).  visible(mc1gt_2,193). happensAt(walking(mc1gt_2),193).
orientation(mc1gt_1,140,194). coords(mc1gt_1,152,97,194).  visible(mc1gt_1,194). happensAt(walking(mc1gt_1),194). orientation(mc1gt_2,149,194). coords(mc1gt_2,101,117,194).  visible(mc1gt_2,194). happensAt(walking(mc1gt_2),194).
orientation(mc1gt_1,140,195). coords(mc1gt_1,152,97,195).  visible(mc1gt_1,195). happensAt(walking(mc1gt_1),195). orientation(mc1gt_2,148,195). coords(mc1gt_2,101,117,195).  visible(mc1gt_2,195). happensAt(walking(mc1gt_2),195).
orientation(mc1gt_1,140,196). coords(mc1gt_1,153,98,196).  visible(mc1gt_1,196). happensAt(walking(mc1gt_1),196). orientation(mc1gt_2,148,196). coords(mc1gt_2,102,117,196).  visible(mc1gt_2,196). happensAt(walking(mc1gt_2),196).
orientation(mc1gt_1,140,197). coords(mc1gt_1,153,100,197).  visible(mc1gt_1,197). happensAt(walking(mc1gt_1),197). orientation(mc1gt_2,148,197). coords(mc1gt_2,102,117,197).  visible(mc1gt_2,197). happensAt(walking(mc1gt_2),197).
orientation(mc1gt_1,139,198). coords(mc1gt_1,154,101,198).  visible(mc1gt_1,198). happensAt(walking(mc1gt_1),198). orientation(mc1gt_2,147,198). coords(mc1gt_2,102,117,198).  visible(mc1gt_2,198). happensAt(walking(mc1gt_2),198).
orientation(mc1gt_1,139,199). coords(mc1gt_1,155,103,199).  visible(mc1gt_1,199). happensAt(walking(mc1gt_1),199). orientation(mc1gt_2,147,199). coords(mc1gt_2,102,117,199).  visible(mc1gt_2,199). happensAt(walking(mc1gt_2),199).
orientation(mc1gt_1,139,200). coords(mc1gt_1,157,105,200).  visible(mc1gt_1,200). happensAt(walking(mc1gt_1),200). orientation(mc1gt_2,147,200). coords(mc1gt_2,103,118,200).  visible(mc1gt_2,200). happensAt(walking(mc1gt_2),200).
orientation(mc1gt_1,137,201). coords(mc1gt_1,159,106,201).  visible(mc1gt_1,201). happensAt(walking(mc1gt_1),201). orientation(mc1gt_2,147,201). coords(mc1gt_2,103,118,201).  visible(mc1gt_2,201). happensAt(walking(mc1gt_2),201).
orientation(mc1gt_1,137,202). coords(mc1gt_1,160,107,202).  visible(mc1gt_1,202). happensAt(walking(mc1gt_1),202). orientation(mc1gt_2,148,202). coords(mc1gt_2,103,119,202).  visible(mc1gt_2,202). happensAt(walking(mc1gt_2),202).
orientation(mc1gt_1,137,203). coords(mc1gt_1,161,108,203).  visible(mc1gt_1,203). happensAt(walking(mc1gt_1),203). orientation(mc1gt_2,148,203). coords(mc1gt_2,104,120,203).  visible(mc1gt_2,203). happensAt(walking(mc1gt_2),203).
orientation(mc1gt_1,137,204). coords(mc1gt_1,161,109,204).  visible(mc1gt_1,204). happensAt(walking(mc1gt_1),204). orientation(mc1gt_2,148,204). coords(mc1gt_2,106,121,204).  visible(mc1gt_2,204). happensAt(walking(mc1gt_2),204).
orientation(mc1gt_1,137,205). coords(mc1gt_1,162,109,205).  visible(mc1gt_1,205). happensAt(walking(mc1gt_1),205). orientation(mc1gt_2,148,205). coords(mc1gt_2,106,122,205).  visible(mc1gt_2,205). happensAt(walking(mc1gt_2),205).
orientation(mc1gt_1,140,206). coords(mc1gt_1,162,110,206).  visible(mc1gt_1,206). happensAt(walking(mc1gt_1),206). orientation(mc1gt_2,148,206). coords(mc1gt_2,108,122,206).  visible(mc1gt_2,206). happensAt(walking(mc1gt_2),206).
orientation(mc1gt_1,140,207). coords(mc1gt_1,162,110,207).  visible(mc1gt_1,207). happensAt(walking(mc1gt_1),207). orientation(mc1gt_2,148,207). coords(mc1gt_2,109,123,207).  visible(mc1gt_2,207). happensAt(walking(mc1gt_2),207).
orientation(mc1gt_1,140,208). coords(mc1gt_1,162,110,208).  visible(mc1gt_1,208). happensAt(walking(mc1gt_1),208). orientation(mc1gt_2,148,208). coords(mc1gt_2,109,123,208).  visible(mc1gt_2,208). happensAt(walking(mc1gt_2),208).
orientation(mc1gt_1,141,209). coords(mc1gt_1,163,111,209).  visible(mc1gt_1,209). happensAt(walking(mc1gt_1),209). orientation(mc1gt_2,148,209). coords(mc1gt_2,110,124,209).  visible(mc1gt_2,209). happensAt(walking(mc1gt_2),209).
orientation(mc1gt_1,141,210). coords(mc1gt_1,163,111,210).  visible(mc1gt_1,210). happensAt(walking(mc1gt_1),210). orientation(mc1gt_2,148,210). coords(mc1gt_2,110,124,210).  visible(mc1gt_2,210). happensAt(walking(mc1gt_2),210).
orientation(mc1gt_1,143,211). coords(mc1gt_1,163,111,211).  visible(mc1gt_1,211). happensAt(walking(mc1gt_1),211). orientation(mc1gt_2,148,211). coords(mc1gt_2,110,124,211).  visible(mc1gt_2,211). happensAt(walking(mc1gt_2),211).
orientation(mc1gt_1,142,212). coords(mc1gt_1,164,112,212).  visible(mc1gt_1,212). happensAt(walking(mc1gt_1),212). orientation(mc1gt_2,148,212). coords(mc1gt_2,111,124,212).  visible(mc1gt_2,212). happensAt(walking(mc1gt_2),212).
orientation(mc1gt_1,143,213). coords(mc1gt_1,167,112,213).  visible(mc1gt_1,213). happensAt(walking(mc1gt_1),213). orientation(mc1gt_2,148,213). coords(mc1gt_2,111,125,213).  visible(mc1gt_2,213). happensAt(walking(mc1gt_2),213).
#show holdsAt/2.