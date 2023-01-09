
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

	orientation(wk1gt_0,123,87). coords(wk1gt_0,207,138,87).  visible(wk1gt_0,87). happensAt(walking(wk1gt_0),87). orientation(wk1gt_5,0,87). coords(wk1gt_5,31,197,87).  visible(wk1gt_5,87). happensAt(walking(wk1gt_5),87).
orientation(wk1gt_0,125,88). coords(wk1gt_0,207,136,88).  visible(wk1gt_0,88). happensAt(walking(wk1gt_0),88). orientation(wk1gt_5,0,88). coords(wk1gt_5,32,197,88).  visible(wk1gt_5,88). happensAt(walking(wk1gt_5),88).
orientation(wk1gt_0,130,89). coords(wk1gt_0,207,134,89).  visible(wk1gt_0,89). happensAt(walking(wk1gt_0),89). orientation(wk1gt_5,0,89). coords(wk1gt_5,32,196,89).  visible(wk1gt_5,89). happensAt(walking(wk1gt_5),89).
orientation(wk1gt_0,131,90). coords(wk1gt_0,207,131,90).  visible(wk1gt_0,90). happensAt(walking(wk1gt_0),90). orientation(wk1gt_5,0,90). coords(wk1gt_5,32,195,90).  visible(wk1gt_5,90). happensAt(walking(wk1gt_5),90).
orientation(wk1gt_0,131,91). coords(wk1gt_0,207,129,91).  visible(wk1gt_0,91). happensAt(walking(wk1gt_0),91). orientation(wk1gt_5,0,91). coords(wk1gt_5,32,195,91).  visible(wk1gt_5,91). happensAt(walking(wk1gt_5),91).
orientation(wk1gt_0,131,92). coords(wk1gt_0,206,128,92).  visible(wk1gt_0,92). happensAt(walking(wk1gt_0),92). orientation(wk1gt_5,0,92). coords(wk1gt_5,32,194,92).  visible(wk1gt_5,92). happensAt(walking(wk1gt_5),92).
orientation(wk1gt_0,131,93). coords(wk1gt_0,206,127,93).  visible(wk1gt_0,93). happensAt(walking(wk1gt_0),93). orientation(wk1gt_5,0,93). coords(wk1gt_5,32,194,93).  visible(wk1gt_5,93). happensAt(walking(wk1gt_5),93).
orientation(wk1gt_0,131,94). coords(wk1gt_0,206,127,94).  visible(wk1gt_0,94). happensAt(walking(wk1gt_0),94). orientation(wk1gt_5,0,94). coords(wk1gt_5,32,193,94).  visible(wk1gt_5,94). happensAt(walking(wk1gt_5),94).
orientation(wk1gt_0,131,95). coords(wk1gt_0,206,126,95).  visible(wk1gt_0,95). happensAt(walking(wk1gt_0),95). orientation(wk1gt_5,0,95). coords(wk1gt_5,32,192,95).  visible(wk1gt_5,95). happensAt(walking(wk1gt_5),95).
orientation(wk1gt_0,131,96). coords(wk1gt_0,206,126,96).  visible(wk1gt_0,96). happensAt(walking(wk1gt_0),96). orientation(wk1gt_5,0,96). coords(wk1gt_5,32,191,96).  visible(wk1gt_5,96). happensAt(walking(wk1gt_5),96).
orientation(wk1gt_0,131,97). coords(wk1gt_0,206,126,97).  visible(wk1gt_0,97). happensAt(walking(wk1gt_0),97). orientation(wk1gt_5,0,97). coords(wk1gt_5,34,191,97).  visible(wk1gt_5,97). happensAt(walking(wk1gt_5),97).
orientation(wk1gt_0,131,98). coords(wk1gt_0,206,126,98).  visible(wk1gt_0,98). happensAt(walking(wk1gt_0),98). orientation(wk1gt_5,0,98). coords(wk1gt_5,35,189,98).  visible(wk1gt_5,98). happensAt(walking(wk1gt_5),98).
orientation(wk1gt_0,131,99). coords(wk1gt_0,205,125,99).  visible(wk1gt_0,99). happensAt(walking(wk1gt_0),99). orientation(wk1gt_5,0,99). coords(wk1gt_5,35,188,99).  visible(wk1gt_5,99). happensAt(walking(wk1gt_5),99).
orientation(wk1gt_0,131,100). coords(wk1gt_0,205,125,100).  visible(wk1gt_0,100). happensAt(walking(wk1gt_0),100). orientation(wk1gt_5,0,100). coords(wk1gt_5,36,187,100).  visible(wk1gt_5,100). happensAt(walking(wk1gt_5),100).
orientation(wk1gt_0,131,101). coords(wk1gt_0,205,125,101).  visible(wk1gt_0,101). happensAt(active(wk1gt_0),101). orientation(wk1gt_5,0,101). coords(wk1gt_5,37,186,101).  visible(wk1gt_5,101). happensAt(walking(wk1gt_5),101).
orientation(wk1gt_0,131,102). coords(wk1gt_0,204,125,102).  visible(wk1gt_0,102). happensAt(active(wk1gt_0),102). orientation(wk1gt_5,0,102). coords(wk1gt_5,37,185,102).  visible(wk1gt_5,102). happensAt(walking(wk1gt_5),102).
orientation(wk1gt_0,131,103). coords(wk1gt_0,204,125,103).  visible(wk1gt_0,103). happensAt(active(wk1gt_0),103). orientation(wk1gt_5,0,103). coords(wk1gt_5,37,183,103).  visible(wk1gt_5,103). happensAt(walking(wk1gt_5),103).
orientation(wk1gt_0,131,104). coords(wk1gt_0,204,125,104).  visible(wk1gt_0,104). happensAt(active(wk1gt_0),104). orientation(wk1gt_5,0,104). coords(wk1gt_5,37,183,104).  visible(wk1gt_5,104). happensAt(walking(wk1gt_5),104).
orientation(wk1gt_0,131,105). coords(wk1gt_0,204,125,105).  visible(wk1gt_0,105). happensAt(active(wk1gt_0),105). orientation(wk1gt_5,0,105). coords(wk1gt_5,37,183,105).  visible(wk1gt_5,105). happensAt(walking(wk1gt_5),105).
orientation(wk1gt_0,131,106). coords(wk1gt_0,204,125,106).  visible(wk1gt_0,106). happensAt(active(wk1gt_0),106). orientation(wk1gt_5,0,106). coords(wk1gt_5,37,181,106).  visible(wk1gt_5,106). happensAt(walking(wk1gt_5),106).
orientation(wk1gt_0,131,107). coords(wk1gt_0,204,124,107).  visible(wk1gt_0,107). happensAt(active(wk1gt_0),107). orientation(wk1gt_5,0,107). coords(wk1gt_5,38,181,107).  visible(wk1gt_5,107). happensAt(walking(wk1gt_5),107).
orientation(wk1gt_0,131,108). coords(wk1gt_0,204,124,108).  visible(wk1gt_0,108). happensAt(active(wk1gt_0),108). orientation(wk1gt_5,0,108). coords(wk1gt_5,38,179,108).  visible(wk1gt_5,108). happensAt(walking(wk1gt_5),108).
orientation(wk1gt_0,131,109). coords(wk1gt_0,204,124,109).  visible(wk1gt_0,109). happensAt(active(wk1gt_0),109). orientation(wk1gt_5,0,109). coords(wk1gt_5,39,179,109).  visible(wk1gt_5,109). happensAt(walking(wk1gt_5),109).
orientation(wk1gt_0,131,110). coords(wk1gt_0,204,124,110).  visible(wk1gt_0,110). happensAt(active(wk1gt_0),110). orientation(wk1gt_5,0,110). coords(wk1gt_5,39,179,110).  visible(wk1gt_5,110). happensAt(walking(wk1gt_5),110).
orientation(wk1gt_0,131,111). coords(wk1gt_0,204,124,111).  visible(wk1gt_0,111). happensAt(active(wk1gt_0),111). orientation(wk1gt_5,0,111). coords(wk1gt_5,39,179,111).  visible(wk1gt_5,111). happensAt(walking(wk1gt_5),111).
#show holdsAt/2.