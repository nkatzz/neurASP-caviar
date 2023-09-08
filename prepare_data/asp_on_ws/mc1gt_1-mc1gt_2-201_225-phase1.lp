
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
orientation(mc1gt_1,143,214). coords(mc1gt_1,169,113,214).  visible(mc1gt_1,214). happensAt(walking(mc1gt_1),214). orientation(mc1gt_2,148,214). coords(mc1gt_2,111,125,214).  visible(mc1gt_2,214). happensAt(walking(mc1gt_2),214).
orientation(mc1gt_1,143,215). coords(mc1gt_1,171,114,215).  visible(mc1gt_1,215). happensAt(walking(mc1gt_1),215). orientation(mc1gt_2,152,215). coords(mc1gt_2,113,125,215).  visible(mc1gt_2,215). happensAt(walking(mc1gt_2),215).
orientation(mc1gt_1,142,216). coords(mc1gt_1,173,115,216).  visible(mc1gt_1,216). happensAt(walking(mc1gt_1),216). orientation(mc1gt_2,152,216). coords(mc1gt_2,115,125,216).  visible(mc1gt_2,216). happensAt(walking(mc1gt_2),216).
orientation(mc1gt_1,141,217). coords(mc1gt_1,174,116,217).  visible(mc1gt_1,217). happensAt(walking(mc1gt_1),217). orientation(mc1gt_2,152,217). coords(mc1gt_2,116,125,217).  visible(mc1gt_2,217). happensAt(walking(mc1gt_2),217).
orientation(mc1gt_1,141,218). coords(mc1gt_1,175,117,218).  visible(mc1gt_1,218). happensAt(walking(mc1gt_1),218). orientation(mc1gt_2,152,218). coords(mc1gt_2,117,125,218).  visible(mc1gt_2,218). happensAt(walking(mc1gt_2),218).
orientation(mc1gt_1,141,219). coords(mc1gt_1,176,118,219).  visible(mc1gt_1,219). happensAt(walking(mc1gt_1),219). orientation(mc1gt_2,152,219). coords(mc1gt_2,119,126,219).  visible(mc1gt_2,219). happensAt(walking(mc1gt_2),219).
orientation(mc1gt_1,141,220). coords(mc1gt_1,177,119,220).  visible(mc1gt_1,220). happensAt(walking(mc1gt_1),220). orientation(mc1gt_2,152,220). coords(mc1gt_2,119,126,220).  visible(mc1gt_2,220). happensAt(walking(mc1gt_2),220).
orientation(mc1gt_1,139,221). coords(mc1gt_1,178,119,221).  visible(mc1gt_1,221). happensAt(walking(mc1gt_1),221). orientation(mc1gt_2,152,221). coords(mc1gt_2,121,127,221).  visible(mc1gt_2,221). happensAt(walking(mc1gt_2),221).
orientation(mc1gt_1,138,222). coords(mc1gt_1,178,120,222).  visible(mc1gt_1,222). happensAt(walking(mc1gt_1),222). orientation(mc1gt_2,152,222). coords(mc1gt_2,121,127,222).  visible(mc1gt_2,222). happensAt(walking(mc1gt_2),222).
orientation(mc1gt_1,138,223). coords(mc1gt_1,178,120,223).  visible(mc1gt_1,223). happensAt(walking(mc1gt_1),223). orientation(mc1gt_2,152,223). coords(mc1gt_2,122,127,223).  visible(mc1gt_2,223). happensAt(walking(mc1gt_2),223).
orientation(mc1gt_1,138,224). coords(mc1gt_1,179,120,224).  visible(mc1gt_1,224). happensAt(walking(mc1gt_1),224). orientation(mc1gt_2,149,224). coords(mc1gt_2,122,127,224).  visible(mc1gt_2,224). happensAt(walking(mc1gt_2),224).
orientation(mc1gt_1,139,225). coords(mc1gt_1,179,121,225).  visible(mc1gt_1,225). happensAt(walking(mc1gt_1),225). orientation(mc1gt_2,148,225). coords(mc1gt_2,123,127,225).  visible(mc1gt_2,225). happensAt(walking(mc1gt_2),225).
#show holdsAt/2.