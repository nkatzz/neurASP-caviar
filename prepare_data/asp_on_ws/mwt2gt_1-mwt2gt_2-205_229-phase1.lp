
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

	orientation(mwt2gt_1,141,205). coords(mwt2gt_1,152,49,205).  visible(mwt2gt_1,205). happensAt(walking(mwt2gt_1),205). orientation(mwt2gt_2,177,205). coords(mwt2gt_2,241,231,205).  visible(mwt2gt_2,205). happensAt(walking(mwt2gt_2),205).
orientation(mwt2gt_1,141,206). coords(mwt2gt_1,151,49,206).  visible(mwt2gt_1,206). happensAt(walking(mwt2gt_1),206). orientation(mwt2gt_2,177,206). coords(mwt2gt_2,241,229,206).  visible(mwt2gt_2,206). happensAt(walking(mwt2gt_2),206).
orientation(mwt2gt_1,141,207). coords(mwt2gt_1,151,49,207).  visible(mwt2gt_1,207). happensAt(walking(mwt2gt_1),207). orientation(mwt2gt_2,177,207). coords(mwt2gt_2,241,227,207).  visible(mwt2gt_2,207). happensAt(walking(mwt2gt_2),207).
orientation(mwt2gt_1,141,208). coords(mwt2gt_1,150,49,208).  visible(mwt2gt_1,208). happensAt(walking(mwt2gt_1),208). orientation(mwt2gt_2,177,208). coords(mwt2gt_2,240,224,208).  visible(mwt2gt_2,208). happensAt(walking(mwt2gt_2),208).
orientation(mwt2gt_1,141,209). coords(mwt2gt_1,149,49,209).  visible(mwt2gt_1,209). happensAt(walking(mwt2gt_1),209). orientation(mwt2gt_2,177,209). coords(mwt2gt_2,240,222,209).  visible(mwt2gt_2,209). happensAt(walking(mwt2gt_2),209).
orientation(mwt2gt_1,141,210). coords(mwt2gt_1,149,49,210).  visible(mwt2gt_1,210). happensAt(walking(mwt2gt_1),210). orientation(mwt2gt_2,177,210). coords(mwt2gt_2,239,219,210).  visible(mwt2gt_2,210). happensAt(walking(mwt2gt_2),210).
orientation(mwt2gt_1,141,211). coords(mwt2gt_1,147,49,211).  visible(mwt2gt_1,211). happensAt(walking(mwt2gt_1),211). orientation(mwt2gt_2,177,211). coords(mwt2gt_2,239,219,211).  visible(mwt2gt_2,211). happensAt(walking(mwt2gt_2),211).
orientation(mwt2gt_1,141,212). coords(mwt2gt_1,147,49,212).  visible(mwt2gt_1,212). happensAt(walking(mwt2gt_1),212). orientation(mwt2gt_2,174,212). coords(mwt2gt_2,238,219,212).  visible(mwt2gt_2,212). happensAt(walking(mwt2gt_2),212).
orientation(mwt2gt_1,141,213). coords(mwt2gt_1,147,49,213).  visible(mwt2gt_1,213). happensAt(walking(mwt2gt_1),213). orientation(mwt2gt_2,174,213). coords(mwt2gt_2,237,218,213).  visible(mwt2gt_2,213). happensAt(walking(mwt2gt_2),213).
orientation(mwt2gt_1,141,214). coords(mwt2gt_1,147,49,214).  visible(mwt2gt_1,214). happensAt(walking(mwt2gt_1),214). orientation(mwt2gt_2,174,214). coords(mwt2gt_2,235,216,214).  visible(mwt2gt_2,214). happensAt(walking(mwt2gt_2),214).
orientation(mwt2gt_1,141,215). coords(mwt2gt_1,147,49,215).  visible(mwt2gt_1,215). happensAt(walking(mwt2gt_1),215). orientation(mwt2gt_2,174,215). coords(mwt2gt_2,234,214,215).  visible(mwt2gt_2,215). happensAt(walking(mwt2gt_2),215).
orientation(mwt2gt_1,141,216). coords(mwt2gt_1,147,51,216).  visible(mwt2gt_1,216). happensAt(walking(mwt2gt_1),216). orientation(mwt2gt_2,168,216). coords(mwt2gt_2,233,211,216).  visible(mwt2gt_2,216). happensAt(walking(mwt2gt_2),216).
orientation(mwt2gt_1,141,217). coords(mwt2gt_1,147,52,217).  visible(mwt2gt_1,217). happensAt(walking(mwt2gt_1),217). orientation(mwt2gt_2,168,217). coords(mwt2gt_2,231,207,217).  visible(mwt2gt_2,217). happensAt(walking(mwt2gt_2),217).
orientation(mwt2gt_1,141,218). coords(mwt2gt_1,146,54,218).  visible(mwt2gt_1,218). happensAt(walking(mwt2gt_1),218). orientation(mwt2gt_2,168,218). coords(mwt2gt_2,230,204,218).  visible(mwt2gt_2,218). happensAt(walking(mwt2gt_2),218).
orientation(mwt2gt_1,137,219). coords(mwt2gt_1,146,55,219).  visible(mwt2gt_1,219). happensAt(walking(mwt2gt_1),219). orientation(mwt2gt_2,168,219). coords(mwt2gt_2,228,202,219).  visible(mwt2gt_2,219). happensAt(walking(mwt2gt_2),219).
orientation(mwt2gt_1,133,220). coords(mwt2gt_1,146,56,220).  visible(mwt2gt_1,220). happensAt(walking(mwt2gt_1),220). orientation(mwt2gt_2,168,220). coords(mwt2gt_2,227,201,220).  visible(mwt2gt_2,220). happensAt(walking(mwt2gt_2),220).
orientation(mwt2gt_1,133,221). coords(mwt2gt_1,146,57,221).  visible(mwt2gt_1,221). happensAt(walking(mwt2gt_1),221). orientation(mwt2gt_2,168,221). coords(mwt2gt_2,227,198,221).  visible(mwt2gt_2,221). happensAt(walking(mwt2gt_2),221).
orientation(mwt2gt_1,133,222). coords(mwt2gt_1,145,58,222).  visible(mwt2gt_1,222). happensAt(walking(mwt2gt_1),222). orientation(mwt2gt_2,168,222). coords(mwt2gt_2,226,195,222).  visible(mwt2gt_2,222). happensAt(walking(mwt2gt_2),222).
orientation(mwt2gt_1,133,223). coords(mwt2gt_1,146,58,223).  visible(mwt2gt_1,223). happensAt(walking(mwt2gt_1),223). orientation(mwt2gt_2,168,223). coords(mwt2gt_2,226,194,223).  visible(mwt2gt_2,223). happensAt(walking(mwt2gt_2),223).
orientation(mwt2gt_1,133,224). coords(mwt2gt_1,145,59,224).  visible(mwt2gt_1,224). happensAt(walking(mwt2gt_1),224). orientation(mwt2gt_2,159,224). coords(mwt2gt_2,226,194,224).  visible(mwt2gt_2,224). happensAt(walking(mwt2gt_2),224).
orientation(mwt2gt_1,136,225). coords(mwt2gt_1,145,59,225).  visible(mwt2gt_1,225). happensAt(walking(mwt2gt_1),225). orientation(mwt2gt_2,150,225). coords(mwt2gt_2,225,194,225).  visible(mwt2gt_2,225). happensAt(walking(mwt2gt_2),225).
orientation(mwt2gt_1,136,226). coords(mwt2gt_1,144,59,226).  visible(mwt2gt_1,226). happensAt(walking(mwt2gt_1),226). orientation(mwt2gt_2,146,226). coords(mwt2gt_2,224,192,226).  visible(mwt2gt_2,226). happensAt(walking(mwt2gt_2),226).
orientation(mwt2gt_1,142,227). coords(mwt2gt_1,144,60,227).  visible(mwt2gt_1,227). happensAt(walking(mwt2gt_1),227). orientation(mwt2gt_2,146,227). coords(mwt2gt_2,223,190,227).  visible(mwt2gt_2,227). happensAt(walking(mwt2gt_2),227).
orientation(mwt2gt_1,142,228). coords(mwt2gt_1,144,60,228).  visible(mwt2gt_1,228). happensAt(walking(mwt2gt_1),228). orientation(mwt2gt_2,146,228). coords(mwt2gt_2,221,187,228).  visible(mwt2gt_2,228). happensAt(walking(mwt2gt_2),228).
orientation(mwt2gt_1,142,229). coords(mwt2gt_1,143,60,229).  visible(mwt2gt_1,229). happensAt(walking(mwt2gt_1),229). orientation(mwt2gt_2,146,229). coords(mwt2gt_2,220,184,229).  visible(mwt2gt_2,229). happensAt(walking(mwt2gt_2),229).
#show holdsAt/2.