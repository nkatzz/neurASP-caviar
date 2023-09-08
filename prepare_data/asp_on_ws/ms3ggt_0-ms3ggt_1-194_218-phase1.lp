
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

	orientation(ms3ggt_0,157,194). coords(ms3ggt_0,63,142,194).  visible(ms3ggt_0,194). happensAt(walking(ms3ggt_0),194). orientation(ms3ggt_1,154,194). coords(ms3ggt_1,37,164,194).  visible(ms3ggt_1,194). happensAt(walking(ms3ggt_1),194).
orientation(ms3ggt_0,157,195). coords(ms3ggt_0,62,143,195).  visible(ms3ggt_0,195). happensAt(walking(ms3ggt_0),195). orientation(ms3ggt_1,154,195). coords(ms3ggt_1,37,164,195).  visible(ms3ggt_1,195). happensAt(walking(ms3ggt_1),195).
orientation(ms3ggt_0,157,196). coords(ms3ggt_0,62,143,196).  visible(ms3ggt_0,196). happensAt(walking(ms3ggt_0),196). orientation(ms3ggt_1,154,196). coords(ms3ggt_1,37,164,196).  visible(ms3ggt_1,196). happensAt(walking(ms3ggt_1),196).
orientation(ms3ggt_0,157,197). coords(ms3ggt_0,62,144,197).  visible(ms3ggt_0,197). happensAt(walking(ms3ggt_0),197). orientation(ms3ggt_1,154,197). coords(ms3ggt_1,37,164,197).  visible(ms3ggt_1,197). happensAt(walking(ms3ggt_1),197).
orientation(ms3ggt_0,145,198). coords(ms3ggt_0,61,148,198).  visible(ms3ggt_0,198). happensAt(walking(ms3ggt_0),198). orientation(ms3ggt_1,154,198). coords(ms3ggt_1,37,164,198).  visible(ms3ggt_1,198). happensAt(walking(ms3ggt_1),198).
orientation(ms3ggt_0,145,199). coords(ms3ggt_0,61,148,199).  visible(ms3ggt_0,199). happensAt(walking(ms3ggt_0),199). orientation(ms3ggt_1,154,199). coords(ms3ggt_1,37,164,199).  visible(ms3ggt_1,199). happensAt(walking(ms3ggt_1),199).
orientation(ms3ggt_0,145,200). coords(ms3ggt_0,61,148,200).  visible(ms3ggt_0,200). happensAt(walking(ms3ggt_0),200). orientation(ms3ggt_1,154,200). coords(ms3ggt_1,37,164,200).  visible(ms3ggt_1,200). happensAt(walking(ms3ggt_1),200).
orientation(ms3ggt_0,145,201). coords(ms3ggt_0,61,148,201).  visible(ms3ggt_0,201). happensAt(walking(ms3ggt_0),201). orientation(ms3ggt_1,154,201). coords(ms3ggt_1,37,164,201).  visible(ms3ggt_1,201). happensAt(walking(ms3ggt_1),201).
orientation(ms3ggt_0,145,202). coords(ms3ggt_0,61,148,202).  visible(ms3ggt_0,202). happensAt(walking(ms3ggt_0),202). orientation(ms3ggt_1,154,202). coords(ms3ggt_1,37,164,202).  visible(ms3ggt_1,202). happensAt(walking(ms3ggt_1),202).
orientation(ms3ggt_0,145,203). coords(ms3ggt_0,61,150,203).  visible(ms3ggt_0,203). happensAt(walking(ms3ggt_0),203). orientation(ms3ggt_1,154,203). coords(ms3ggt_1,37,164,203).  visible(ms3ggt_1,203). happensAt(walking(ms3ggt_1),203).
orientation(ms3ggt_0,145,204). coords(ms3ggt_0,61,151,204).  visible(ms3ggt_0,204). happensAt(walking(ms3ggt_0),204). orientation(ms3ggt_1,154,204). coords(ms3ggt_1,38,165,204).  visible(ms3ggt_1,204). happensAt(walking(ms3ggt_1),204).
orientation(ms3ggt_0,145,205). coords(ms3ggt_0,61,151,205).  visible(ms3ggt_0,205). happensAt(walking(ms3ggt_0),205). orientation(ms3ggt_1,154,205). coords(ms3ggt_1,39,165,205).  visible(ms3ggt_1,205). happensAt(walking(ms3ggt_1),205).
orientation(ms3ggt_0,145,206). coords(ms3ggt_0,60,152,206).  visible(ms3ggt_0,206). happensAt(walking(ms3ggt_0),206). orientation(ms3ggt_1,154,206). coords(ms3ggt_1,41,165,206).  visible(ms3ggt_1,206). happensAt(walking(ms3ggt_1),206).
orientation(ms3ggt_0,145,207). coords(ms3ggt_0,60,152,207).  visible(ms3ggt_0,207). happensAt(walking(ms3ggt_0),207). orientation(ms3ggt_1,154,207). coords(ms3ggt_1,41,165,207).  visible(ms3ggt_1,207). happensAt(walking(ms3ggt_1),207).
orientation(ms3ggt_0,145,208). coords(ms3ggt_0,60,152,208).  visible(ms3ggt_0,208). happensAt(walking(ms3ggt_0),208). orientation(ms3ggt_1,154,208). coords(ms3ggt_1,41,165,208).  visible(ms3ggt_1,208). happensAt(walking(ms3ggt_1),208).
orientation(ms3ggt_0,145,209). coords(ms3ggt_0,60,153,209).  visible(ms3ggt_0,209). happensAt(walking(ms3ggt_0),209). orientation(ms3ggt_1,154,209). coords(ms3ggt_1,41,165,209).  visible(ms3ggt_1,209). happensAt(walking(ms3ggt_1),209).
orientation(ms3ggt_0,145,210). coords(ms3ggt_0,59,153,210).  visible(ms3ggt_0,210). happensAt(walking(ms3ggt_0),210). orientation(ms3ggt_1,154,210). coords(ms3ggt_1,42,165,210).  visible(ms3ggt_1,210). happensAt(walking(ms3ggt_1),210).
orientation(ms3ggt_0,145,211). coords(ms3ggt_0,59,154,211).  visible(ms3ggt_0,211). happensAt(walking(ms3ggt_0),211). orientation(ms3ggt_1,154,211). coords(ms3ggt_1,42,165,211).  visible(ms3ggt_1,211). happensAt(walking(ms3ggt_1),211).
orientation(ms3ggt_0,150,212). coords(ms3ggt_0,59,153,212).  visible(ms3ggt_0,212). happensAt(walking(ms3ggt_0),212). orientation(ms3ggt_1,159,212). coords(ms3ggt_1,42,164,212).  visible(ms3ggt_1,212). happensAt(walking(ms3ggt_1),212).
orientation(ms3ggt_0,154,213). coords(ms3ggt_0,58,153,213).  visible(ms3ggt_0,213). happensAt(walking(ms3ggt_0),213). orientation(ms3ggt_1,159,213). coords(ms3ggt_1,44,165,213).  visible(ms3ggt_1,213). happensAt(walking(ms3ggt_1),213).
orientation(ms3ggt_0,154,214). coords(ms3ggt_0,58,153,214).  visible(ms3ggt_0,214). happensAt(walking(ms3ggt_0),214). orientation(ms3ggt_1,159,214). coords(ms3ggt_1,44,165,214).  visible(ms3ggt_1,214). happensAt(walking(ms3ggt_1),214).
orientation(ms3ggt_0,154,215). coords(ms3ggt_0,58,153,215).  visible(ms3ggt_0,215). happensAt(walking(ms3ggt_0),215). orientation(ms3ggt_1,159,215). coords(ms3ggt_1,44,165,215).  visible(ms3ggt_1,215). happensAt(walking(ms3ggt_1),215).
orientation(ms3ggt_0,154,216). coords(ms3ggt_0,58,153,216).  visible(ms3ggt_0,216). happensAt(walking(ms3ggt_0),216). orientation(ms3ggt_1,159,216). coords(ms3ggt_1,44,165,216).  visible(ms3ggt_1,216). happensAt(walking(ms3ggt_1),216).
orientation(ms3ggt_0,154,217). coords(ms3ggt_0,58,154,217).  visible(ms3ggt_0,217). happensAt(walking(ms3ggt_0),217). orientation(ms3ggt_1,159,217). coords(ms3ggt_1,44,165,217).  visible(ms3ggt_1,217). happensAt(walking(ms3ggt_1),217).
orientation(ms3ggt_0,154,218). coords(ms3ggt_0,58,155,218).  visible(ms3ggt_0,218). happensAt(walking(ms3ggt_0),218). orientation(ms3ggt_1,159,218). coords(ms3ggt_1,44,165,218).  visible(ms3ggt_1,218). happensAt(walking(ms3ggt_1),218).
#show holdsAt/2.