
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

	orientation(wk1gt_0,123,87). coords(wk1gt_0,207,138,87).  visible(wk1gt_0,87). happensAt(walking(wk1gt_0),87). orientation(wk1gt_4,0,87). coords(wk1gt_4,39,195,87).  visible(wk1gt_4,87). happensAt(walking(wk1gt_4),87).
orientation(wk1gt_0,125,88). coords(wk1gt_0,207,136,88).  visible(wk1gt_0,88). happensAt(walking(wk1gt_0),88). orientation(wk1gt_4,0,88). coords(wk1gt_4,39,194,88).  visible(wk1gt_4,88). happensAt(walking(wk1gt_4),88).
orientation(wk1gt_0,130,89). coords(wk1gt_0,207,134,89).  visible(wk1gt_0,89). happensAt(walking(wk1gt_0),89). orientation(wk1gt_4,0,89). coords(wk1gt_4,39,194,89).  visible(wk1gt_4,89). happensAt(walking(wk1gt_4),89).
orientation(wk1gt_0,131,90). coords(wk1gt_0,207,131,90).  visible(wk1gt_0,90). happensAt(walking(wk1gt_0),90). orientation(wk1gt_4,0,90). coords(wk1gt_4,39,194,90).  visible(wk1gt_4,90). happensAt(walking(wk1gt_4),90).
orientation(wk1gt_0,131,91). coords(wk1gt_0,207,129,91).  visible(wk1gt_0,91). happensAt(walking(wk1gt_0),91). orientation(wk1gt_4,0,91). coords(wk1gt_4,39,194,91).  visible(wk1gt_4,91). happensAt(walking(wk1gt_4),91).
orientation(wk1gt_0,131,92). coords(wk1gt_0,206,128,92).  visible(wk1gt_0,92). happensAt(walking(wk1gt_0),92). orientation(wk1gt_4,0,92). coords(wk1gt_4,39,194,92).  visible(wk1gt_4,92). happensAt(walking(wk1gt_4),92).
orientation(wk1gt_0,131,93). coords(wk1gt_0,206,127,93).  visible(wk1gt_0,93). happensAt(walking(wk1gt_0),93). orientation(wk1gt_4,0,93). coords(wk1gt_4,39,194,93).  visible(wk1gt_4,93). happensAt(walking(wk1gt_4),93).
orientation(wk1gt_0,131,94). coords(wk1gt_0,206,127,94).  visible(wk1gt_0,94). happensAt(walking(wk1gt_0),94). orientation(wk1gt_4,0,94). coords(wk1gt_4,39,194,94).  visible(wk1gt_4,94). happensAt(walking(wk1gt_4),94).
orientation(wk1gt_0,131,95). coords(wk1gt_0,206,126,95).  visible(wk1gt_0,95). happensAt(walking(wk1gt_0),95). orientation(wk1gt_4,0,95). coords(wk1gt_4,39,194,95).  visible(wk1gt_4,95). happensAt(walking(wk1gt_4),95).
orientation(wk1gt_0,131,96). coords(wk1gt_0,206,126,96).  visible(wk1gt_0,96). happensAt(walking(wk1gt_0),96). orientation(wk1gt_4,0,96). coords(wk1gt_4,39,194,96).  visible(wk1gt_4,96). happensAt(walking(wk1gt_4),96).
orientation(wk1gt_0,131,97). coords(wk1gt_0,206,126,97).  visible(wk1gt_0,97). happensAt(walking(wk1gt_0),97). orientation(wk1gt_4,0,97). coords(wk1gt_4,41,194,97).  visible(wk1gt_4,97). happensAt(walking(wk1gt_4),97).
orientation(wk1gt_0,131,98). coords(wk1gt_0,206,126,98).  visible(wk1gt_0,98). happensAt(walking(wk1gt_0),98). orientation(wk1gt_4,0,98). coords(wk1gt_4,41,194,98).  visible(wk1gt_4,98). happensAt(walking(wk1gt_4),98).
orientation(wk1gt_0,131,99). coords(wk1gt_0,205,125,99).  visible(wk1gt_0,99). happensAt(walking(wk1gt_0),99). orientation(wk1gt_4,0,99). coords(wk1gt_4,41,193,99).  visible(wk1gt_4,99). happensAt(walking(wk1gt_4),99).
orientation(wk1gt_0,131,100). coords(wk1gt_0,205,125,100).  visible(wk1gt_0,100). happensAt(walking(wk1gt_0),100). orientation(wk1gt_4,0,100). coords(wk1gt_4,41,193,100).  visible(wk1gt_4,100). happensAt(walking(wk1gt_4),100).
orientation(wk1gt_0,131,101). coords(wk1gt_0,205,125,101).  visible(wk1gt_0,101). happensAt(active(wk1gt_0),101). orientation(wk1gt_4,0,101). coords(wk1gt_4,43,191,101).  visible(wk1gt_4,101). happensAt(walking(wk1gt_4),101).
orientation(wk1gt_0,131,102). coords(wk1gt_0,204,125,102).  visible(wk1gt_0,102). happensAt(active(wk1gt_0),102). orientation(wk1gt_4,0,102). coords(wk1gt_4,44,190,102).  visible(wk1gt_4,102). happensAt(walking(wk1gt_4),102).
orientation(wk1gt_0,131,103). coords(wk1gt_0,204,125,103).  visible(wk1gt_0,103). happensAt(active(wk1gt_0),103). orientation(wk1gt_4,0,103). coords(wk1gt_4,44,190,103).  visible(wk1gt_4,103). happensAt(walking(wk1gt_4),103).
orientation(wk1gt_0,131,104). coords(wk1gt_0,204,125,104).  visible(wk1gt_0,104). happensAt(active(wk1gt_0),104). orientation(wk1gt_4,0,104). coords(wk1gt_4,44,190,104).  visible(wk1gt_4,104). happensAt(walking(wk1gt_4),104).
orientation(wk1gt_0,131,105). coords(wk1gt_0,204,125,105).  visible(wk1gt_0,105). happensAt(active(wk1gt_0),105). orientation(wk1gt_4,0,105). coords(wk1gt_4,44,189,105).  visible(wk1gt_4,105). happensAt(walking(wk1gt_4),105).
orientation(wk1gt_0,131,106). coords(wk1gt_0,204,125,106).  visible(wk1gt_0,106). happensAt(active(wk1gt_0),106). orientation(wk1gt_4,0,106). coords(wk1gt_4,44,189,106).  visible(wk1gt_4,106). happensAt(walking(wk1gt_4),106).
orientation(wk1gt_0,131,107). coords(wk1gt_0,204,124,107).  visible(wk1gt_0,107). happensAt(active(wk1gt_0),107). orientation(wk1gt_4,0,107). coords(wk1gt_4,44,188,107).  visible(wk1gt_4,107). happensAt(walking(wk1gt_4),107).
orientation(wk1gt_0,131,108). coords(wk1gt_0,204,124,108).  visible(wk1gt_0,108). happensAt(active(wk1gt_0),108). orientation(wk1gt_4,0,108). coords(wk1gt_4,44,187,108).  visible(wk1gt_4,108). happensAt(walking(wk1gt_4),108).
orientation(wk1gt_0,131,109). coords(wk1gt_0,204,124,109).  visible(wk1gt_0,109). happensAt(active(wk1gt_0),109). orientation(wk1gt_4,0,109). coords(wk1gt_4,44,187,109).  visible(wk1gt_4,109). happensAt(walking(wk1gt_4),109).
orientation(wk1gt_0,131,110). coords(wk1gt_0,204,124,110).  visible(wk1gt_0,110). happensAt(active(wk1gt_0),110). orientation(wk1gt_4,0,110). coords(wk1gt_4,44,187,110).  visible(wk1gt_4,110). happensAt(walking(wk1gt_4),110).
orientation(wk1gt_0,131,111). coords(wk1gt_0,204,124,111).  visible(wk1gt_0,111). happensAt(active(wk1gt_0),111). orientation(wk1gt_4,0,111). coords(wk1gt_4,44,187,111).  visible(wk1gt_4,111). happensAt(walking(wk1gt_4),111).
#show holdsAt/2.