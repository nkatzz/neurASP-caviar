
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

	orientation(rwgt_0,160,87). coords(rwgt_0,255,216,87).  visible(rwgt_0,87). happensAt(walking(rwgt_0),87). orientation(rwgt_1,149,87). coords(rwgt_1,194,177,87).  visible(rwgt_1,87). happensAt(walking(rwgt_1),87).
orientation(rwgt_0,168,88). coords(rwgt_0,252,212,88).  visible(rwgt_0,88). happensAt(walking(rwgt_0),88). orientation(rwgt_1,149,88). coords(rwgt_1,193,173,88).  visible(rwgt_1,88). happensAt(walking(rwgt_1),88).
orientation(rwgt_0,168,89). coords(rwgt_0,249,208,89).  visible(rwgt_0,89). happensAt(walking(rwgt_0),89). orientation(rwgt_1,149,89). coords(rwgt_1,190,170,89).  visible(rwgt_1,89). happensAt(walking(rwgt_1),89).
orientation(rwgt_0,168,90). coords(rwgt_0,246,206,90).  visible(rwgt_0,90). happensAt(walking(rwgt_0),90). orientation(rwgt_1,149,90). coords(rwgt_1,188,167,90).  visible(rwgt_1,90). happensAt(walking(rwgt_1),90).
orientation(rwgt_0,168,91). coords(rwgt_0,243,205,91).  visible(rwgt_0,91). happensAt(walking(rwgt_0),91). orientation(rwgt_1,149,91). coords(rwgt_1,185,164,91).  visible(rwgt_1,91). happensAt(walking(rwgt_1),91).
orientation(rwgt_0,168,92). coords(rwgt_0,243,203,92).  visible(rwgt_0,92). happensAt(walking(rwgt_0),92). orientation(rwgt_1,149,92). coords(rwgt_1,184,162,92).  visible(rwgt_1,92). happensAt(walking(rwgt_1),92).
orientation(rwgt_0,168,93). coords(rwgt_0,242,201,93).  visible(rwgt_0,93). happensAt(walking(rwgt_0),93). orientation(rwgt_1,149,93). coords(rwgt_1,182,161,93).  visible(rwgt_1,93). happensAt(walking(rwgt_1),93).
orientation(rwgt_0,168,94). coords(rwgt_0,241,200,94).  visible(rwgt_0,94). happensAt(walking(rwgt_0),94). orientation(rwgt_1,149,94). coords(rwgt_1,182,160,94).  visible(rwgt_1,94). happensAt(walking(rwgt_1),94).
orientation(rwgt_0,168,95). coords(rwgt_0,241,198,95).  visible(rwgt_0,95). happensAt(walking(rwgt_0),95). orientation(rwgt_1,149,95). coords(rwgt_1,181,159,95).  visible(rwgt_1,95). happensAt(walking(rwgt_1),95).
orientation(rwgt_0,168,96). coords(rwgt_0,240,197,96).  visible(rwgt_0,96). happensAt(walking(rwgt_0),96). orientation(rwgt_1,149,96). coords(rwgt_1,181,158,96).  visible(rwgt_1,96). happensAt(walking(rwgt_1),96).
orientation(rwgt_0,153,97). coords(rwgt_0,239,196,97).  visible(rwgt_0,97). happensAt(walking(rwgt_0),97). orientation(rwgt_1,149,97). coords(rwgt_1,181,156,97).  visible(rwgt_1,97). happensAt(walking(rwgt_1),97).
orientation(rwgt_0,152,98). coords(rwgt_0,237,196,98).  visible(rwgt_0,98). happensAt(walking(rwgt_0),98). orientation(rwgt_1,149,98). coords(rwgt_1,180,155,98).  visible(rwgt_1,98). happensAt(walking(rwgt_1),98).
orientation(rwgt_0,148,99). coords(rwgt_0,235,195,99).  visible(rwgt_0,99). happensAt(walking(rwgt_0),99). orientation(rwgt_1,149,99). coords(rwgt_1,180,153,99).  visible(rwgt_1,99). happensAt(walking(rwgt_1),99).
orientation(rwgt_0,148,100). coords(rwgt_0,233,192,100).  visible(rwgt_0,100). happensAt(walking(rwgt_0),100). orientation(rwgt_1,149,100). coords(rwgt_1,178,151,100).  visible(rwgt_1,100). happensAt(walking(rwgt_1),100).
orientation(rwgt_0,150,101). coords(rwgt_0,231,189,101).  visible(rwgt_0,101). happensAt(walking(rwgt_0),101). orientation(rwgt_1,149,101). coords(rwgt_1,176,149,101).  visible(rwgt_1,101). happensAt(walking(rwgt_1),101).
orientation(rwgt_0,148,102). coords(rwgt_0,228,186,102).  visible(rwgt_0,102). happensAt(walking(rwgt_0),102). orientation(rwgt_1,149,102). coords(rwgt_1,174,147,102).  visible(rwgt_1,102). happensAt(walking(rwgt_1),102).
orientation(rwgt_0,147,103). coords(rwgt_0,226,183,103).  visible(rwgt_0,103). happensAt(walking(rwgt_0),103). orientation(rwgt_1,149,103). coords(rwgt_1,172,144,103).  visible(rwgt_1,103). happensAt(walking(rwgt_1),103).
orientation(rwgt_0,147,104). coords(rwgt_0,224,179,104).  visible(rwgt_0,104). happensAt(walking(rwgt_0),104). orientation(rwgt_1,149,104). coords(rwgt_1,171,140,104).  visible(rwgt_1,104). happensAt(walking(rwgt_1),104).
orientation(rwgt_0,147,105). coords(rwgt_0,223,177,105).  visible(rwgt_0,105). happensAt(walking(rwgt_0),105). orientation(rwgt_1,149,105). coords(rwgt_1,170,138,105).  visible(rwgt_1,105). happensAt(walking(rwgt_1),105).
orientation(rwgt_0,147,106). coords(rwgt_0,223,176,106).  visible(rwgt_0,106). happensAt(walking(rwgt_0),106). orientation(rwgt_1,149,106). coords(rwgt_1,170,135,106).  visible(rwgt_1,106). happensAt(walking(rwgt_1),106).
orientation(rwgt_0,147,107). coords(rwgt_0,222,175,107).  visible(rwgt_0,107). happensAt(walking(rwgt_0),107). orientation(rwgt_1,149,107). coords(rwgt_1,170,135,107).  visible(rwgt_1,107). happensAt(walking(rwgt_1),107).
orientation(rwgt_0,147,108). coords(rwgt_0,221,175,108).  visible(rwgt_0,108). happensAt(walking(rwgt_0),108). orientation(rwgt_1,149,108). coords(rwgt_1,169,134,108).  visible(rwgt_1,108). happensAt(walking(rwgt_1),108).
orientation(rwgt_0,147,109). coords(rwgt_0,221,173,109).  visible(rwgt_0,109). happensAt(walking(rwgt_0),109). orientation(rwgt_1,149,109). coords(rwgt_1,169,133,109).  visible(rwgt_1,109). happensAt(walking(rwgt_1),109).
orientation(rwgt_0,147,110). coords(rwgt_0,220,171,110).  visible(rwgt_0,110). happensAt(walking(rwgt_0),110). orientation(rwgt_1,149,110). coords(rwgt_1,168,133,110).  visible(rwgt_1,110). happensAt(walking(rwgt_1),110).
orientation(rwgt_0,147,111). coords(rwgt_0,219,170,111).  visible(rwgt_0,111). happensAt(walking(rwgt_0),111). orientation(rwgt_1,149,111). coords(rwgt_1,168,132,111).  visible(rwgt_1,111). happensAt(walking(rwgt_1),111).
#show holdsAt/2.